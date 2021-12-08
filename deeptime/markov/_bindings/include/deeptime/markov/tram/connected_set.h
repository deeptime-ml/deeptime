//
// Created by Maaike on 08/12/2021.
//
#pragma once

#include <algorithm>
#include <deeptime/common.h>

namespace deeptime::tram {


using DTraj = np_array<std::int32_t>;
using DTrajs = std::vector<DTraj>;


template<typename dtype>
using SampleListFromTwoStates = std::vector<std::vector<dtype>>;

template<typename dtype>
using OverlapFunction = std::function<bool(SampleListFromTwoStates<dtype>, SampleListFromTwoStates<dtype>, dtype )>;

using stateVector = std::vector<std::size_t>;
using transitionVector = std::tuple<stateVector, stateVector>;


template<typename dtype>
transitionVector getStateTransitions(const DTrajs &ttrajs, const DTrajs &dtrajs,
                    const std::vector<np_array_nfc<dtype>> &biasMatrices,
                    const np_array<std::int32_t> &stateCounts,
                    std::int32_t nThermStates,
                    std::int32_t nConfStates,
                    dtype connectivity_factor,
                    OverlapFunction<dtype> overlapFunction) {
    // List of transitions from each therm/markov state combination i, to therm/markov state combination j
    // The markov/therm state index in unraveled to one dimension, i.e. counts_matrix[k,i] in this representation has index
    // k*n_conf_states + i
    stateVector i_s;
    stateVector j_s;

    // At each markov state i, compute overlap for each combination of two thermodynamic states k and l.
    for (std::size_t i = 0; i < nConfStates; ++i) {
        for (std::size_t k = 0; k < nThermStates; ++k) {
            // therm state must have counts in markov state i
            if (stateCounts.at(k, i) > 0) {
                for (std::size_t l = 0; l < nThermStates; ++l) {
                    // therm state must have counts in markov state i
                    if (k != l && stateCounts.at(l, i) > 0) {
                        // TODO: we are now looping through the trajectories for each combination of states k/l
                        // this is a waste!
                        // also: if replica exchange pre-processing is done, we can assume that ttraj[i] == i for all i.
                        // So: then we only have to loop through one trajectory per state, and we can do this at the
                        // start of the loop over the markov states: find markov index i in each trajectory and store
                        // the indices in an array of n_therm_states vectors.

                        // Then we can get rid of ttrajs alltogether and life will be good.
                        auto &[samplesFromK, samplesFromL] = getSamplesFromStates(i, k, l, ttrajs, dtrajs, biasMatrices);

                        if (overlapFunction(samplesFromK, samplesFromL, connectivity_factor)) {
                            // push the ravelled index of the therm state transition to the transition list.
                            auto x = i + k * nConfStates;
                            auto y = i + l * nConfStates;
                            i_s.push_back(x);
                            j_s.push_back(y);
                        }
                    }
                }
            }
        }
    }
    return std::tuple(i_s, j_s);
}

template<typename dtype>
std::tuple<SampleListFromTwoStates<dtype>, SampleListFromTwoStates<dtype>> getSamplesFromStates(
        std::size_t i, std::size_t k, std::size_t l, const DTrajs &ttrajs, const DTrajs &dtrajs,
        const std::vector<np_array_nfc<dtype>> &biasMatrices) {

    // find all bias energy values for all samples in markov state i sampled at state k and l.
    // store them in two vectors, depending on the thermodynamic state they were sampled at.
    SampleListFromTwoStates<dtype> samplesFromK;
    SampleListFromTwoStates<dtype> samplesFromL;

    for (std::size_t j = 0; j < dtrajs.size(); ++j) {
        std::size_t trajLength = dtrajs[j].size();

        auto dtraj = dtrajs[j].template unchecked<1>();
        auto ttraj = ttrajs[j].template unchecked<1>();

        for (std::size_t n = 0; n < trajLength; ++n) {
            if (dtraj[n] == i && ttraj[n] == k) {
                samplesFromK.push_back({biasMatrices[j].at(n, k), biasMatrices[j].at(n, l)});
            }
            if (dtraj[n] == i && ttraj[n] == l) {
                samplesFromL.push_back({biasMatrices[j].at(n, k), biasMatrices[j].at(n, l)});
            }
        }
    }
    return std::make_tuple(samplesFromK, samplesFromL);
}


template<typename dtype>
bool hasOverlapPostHocReplicaExchange(SampleListFromTwoStates<dtype> &samplesFromK,
                                      SampleListFromTwoStates<dtype> &samplesFromL,
                                      dtype connectivity_factor) {
    dtype delta = 0;
    dtype n_sum = 0;

    // now compute the overlap between the samples in both vectors
    for (auto el_K : samplesFromK) {
        for (auto el_L: samplesFromL) {
            delta = el_K[0] + el_L[1] - el_K[1] - el_L[0];
            n_sum += std::min(std::exp(delta), 1.0);
        }
    }
    auto n = samplesFromK.size();
    auto m = samplesFromL.size();

    dtype n_avg = n_sum / (n * m);
    return (n + m) * n_avg * connectivity_factor >= 1.0;
}


template<typename dtype>
bool hasOverlapBarVariance(SampleListFromTwoStates<dtype> &samplesFromK,
                           SampleListFromTwoStates<dtype> &samplesFromL,
                           dtype connectivity_factor) {
    auto n = samplesFromK.size();
    auto m = samplesFromL.size();
    dtype *db_IJ = new dtype[n];
    dtype *db_JI = new dtype[m];
    dtype *du = new dtype[n + m];

    // now compute the overlap between the samples in both vectors
    for (int i =0; i < n; ++i) {
        db_IJ[i] = samplesFromK[i][1] - samplesFromK[i][0];
        du[i] = db_IJ[i];
    }
    for (int i = 0; i < m; ++i) {
        db_JI[i] = samplesFromL[i][0] - samplesFromL[i][1];
        du[n + i] = -db_JI[i];
    }
    auto scratch = np_array_nfc<dtype>(std::vector<std::size_t>{m+n});
    auto df = _bar_df(db_IJ, n, db_JI, m, scratch);

    dtype b = 0;
    for(int i = 0; i < n + m; ++i) {
        b += (1.0 / (2.0 + 2.0 * std::cosh(df - du[i] - std::log(1.0 * n/m))));
    }
    return (1 / b - (n+m) / (n * m) ) < connectivity_factor;
}

template<typename dtype>
extern dtype
_bar_df(dtype db_IJ[], std::size_t L1, dtype db_JI[], std::size_t L2,
        np_array_nfc<dtype> scratch) {

    py::buffer_info scratch_buf = scratch.request();
    auto *scratch_ptr = (dtype *) scratch_buf.ptr;

    std::int32_t i;
    dtype ln_avg1;
    dtype ln_avg2;
    for (i = 0; i < L1; i++) {
        scratch_ptr[i] = db_IJ[i] > 0 ? 0 : db_IJ[i];
    }
    ln_avg1 = numeric::kahan::logsumexp_sort_kahan_inplace(scratch_ptr, scratch_ptr + L1);
    for (i = 0; i < L1; i++) {
        scratch_ptr[i] = db_JI[i] > 0 ? 0 : db_JI[i];
    }
    ln_avg2 = numeric::kahan::logsumexp_sort_kahan_inplace(scratch_ptr, scratch_ptr + L2);
    return ln_avg2 - ln_avg1;
}
}
