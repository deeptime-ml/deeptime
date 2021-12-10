//
// Created by Maaike on 08/12/2021.
//
#pragma once

#include <algorithm>
#include <deeptime/common.h>

namespace deeptime::tram {


using DTraj = np_array<std::int32_t>;
using DTrajs = std::vector<DTraj>;

using IndexList = std::vector<std::vector<std::tuple<std::int32_t, std::int32_t>>>;

template<typename dtype>
using OverlapFunction = std::function<bool(std::size_t, std::size_t, IndexList &,
        const std::vector<np_array_nfc<dtype>> &, dtype connectivity_factor)>;

using stateVector = std::vector<std::size_t>;
using transitionVector = std::tuple<stateVector, stateVector>;


template<typename dtype>
transitionVector getStateTransitions(const std::optional<DTrajs> &ttrajs, const DTrajs &dtrajs,
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

        // Get all indices in all trajectories of all samples that were binned in markov state i.
        IndexList sampleIndicesIn_i = getIndexOfSamplesInMarkovState(i, ttrajs, dtrajs, nThermStates);

        for (std::size_t k = 0; k < nThermStates; ++k) {
            // therm state must have counts in markov state i
            if (stateCounts.at(k, i) > 0) {
                for (std::size_t l = 0; l < nThermStates; ++l) {
                    // therm state must have counts in markov state i
                    if (k != l && stateCounts.at(l, i) > 0) {
                        if (overlapFunction(k, l, sampleIndicesIn_i, biasMatrices, connectivity_factor)) {
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

IndexList getIndexOfSamplesInMarkovState(size_t i, const DTrajs &dtrajs, int32_t nThermStates) {
    IndexList indices (nThermStates);

    for (std::size_t j = 0; j < dtrajs.size(); ++j) {
        std::size_t trajLength = dtrajs[j].size();

        auto dtraj = dtrajs[j].template unchecked<1>();

        for (std::int32_t n = 0; n < trajLength; ++n) {
            if (dtraj[n] == i) {
                // markov state i sampled in therm state j can be found at bias matrix index (j, n,)
                indices[j].push_back({j, n});
            }
        }
    }
    return indices;
}

IndexList getIndexOfSamplesInMarkovState(std::size_t i, const DTrajs &ttrajs, const DTrajs &dtrajs,
                                         std::int32_t nThermStates) {
    IndexList indices (nThermStates);

    for (std::size_t j = 0; j < dtrajs.size(); ++j) {
        std::size_t trajLength = dtrajs[j].size();

        auto dtraj = dtrajs[j].template unchecked<1>();
        auto ttraj = ttrajs[j].template unchecked<1>();

        for (std::int32_t n = 0; n < trajLength; ++n) {
            if (dtraj[n] == i) {
                auto k = ttrajs[j].at(n);
                // markov state i sampled in therm state k can be found at bias matrix index (j, n,)
                indices[k].push_back({j, n});
            }
        }
    }
    return indices;
}

IndexList getIndexOfSamplesInMarkovState (std::size_t i, const std::optional<DTrajs> &ttrajs, const DTrajs &dtrajs,
                                          std::int32_t nThermStates) {
    if (ttrajs) {
        return getIndexOfSamplesInMarkovState(i, *ttrajs, dtrajs, nThermStates);
    }
    else {
        return getIndexOfSamplesInMarkovState(i, dtrajs, nThermStates);
    }
}


template<typename dtype>
bool hasOverlapPostHocReplicaExchange(std::size_t k, std::size_t l, IndexList &sampleIndicesIn_i,
                                      const std::vector<np_array_nfc<dtype>> &biasMatrices,
                                      dtype connectivity_factor) {
    dtype delta = 0;
    dtype n_sum = 0;

    // now compute the overlap between the samples in both vectors
    for (auto &[k_j, k_n] : sampleIndicesIn_i[k]) {
        for (auto &[l_j, l_n]: sampleIndicesIn_i[l]) {
            delta = biasMatrices[k_j].at(k_n, k) + biasMatrices[l_j].at(l_n, l)
                    - biasMatrices[k_j].at(k_n, l) - biasMatrices[l_j].at(l_n, k);
            n_sum += std::min(std::exp(delta), 1.0);
        }
    }
    auto n = sampleIndicesIn_i[k].size();
    auto m = sampleIndicesIn_i[l].size();

    dtype n_avg = n_sum / (n * m);
    return (n + m) * n_avg * connectivity_factor >= 1.0;
}


template<typename dtype>
bool hasOverlapBarVariance(std::size_t k, std::size_t l, IndexList &sampleIndicesIn_i,
                           const std::vector<np_array_nfc<dtype>> &biasMatrices,
                            dtype connectivity_factor) {
    auto n = sampleIndicesIn_i[k].size();
    auto m = sampleIndicesIn_i[l].size();
    dtype *db_IJ = new dtype[n];
    dtype *db_JI = new dtype[m];
    dtype *du = new dtype[n + m];

    // now compute the overlap between the samples in both vectors
    for (int i =0; i < n; ++i) {
        auto &[k_j, k_n] = sampleIndicesIn_i[k][i];
        db_IJ[i] = biasMatrices[k_j].at(k_n, l) - biasMatrices[k_j].at(k_n, k);
        du[i] = db_IJ[i];
    }
    for (int i = 0; i < m; ++i) {
        auto &[l_j, l_n] = sampleIndicesIn_i[l][i];
        db_JI[i] = biasMatrices[l_j].at(l_n, k) - biasMatrices[l_j].at(l_n, l);
        du[n + i] = -db_JI[i];
    }
    auto scratch = np_array_nfc<dtype>(std::vector<std::size_t>{m+n});
    auto df = _bar_df(db_IJ, n, db_JI, m, scratch);

    dtype b = 0;
    for(int i = 0; i < n + m; ++i) {
        b += (1.0 / (2.0 + 2.0 * std::cosh(df - du[i] - std::log(1.0 * static_cast<dtype>(n/m)))));
    }
    return 1 / b - ( n + m ) / static_cast<dtype>(n * m) < connectivity_factor;
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
