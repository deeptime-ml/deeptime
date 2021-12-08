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
using OverlapFunction = std::function<bool(int i, int k, int l, const DTrajs &, const DTrajs &,
                                           const std::vector<np_array_nfc<dtype>> &, dtype )>;

template<typename dtype>
std::tuple<std::vector<int32_t>, std::vector<int32_t>>
getStateTransitions(const DTrajs &ttrajs, const DTrajs &dtrajs, const std::vector<np_array_nfc<dtype>> &biasMatrices,
                    const np_array<std::int32_t> &stateCounts,
                    std::int32_t nThermStates,
                    std::int32_t nConfStates,
                    dtype connectivity_factor,
                    OverlapFunction<dtype> overlapFunction) {
    // List of transitions from each therm/markov state combination i, to therm/markov state combination j
    // The markov/therm state index in unraveled to one dimension, i.e. counts_matrix[k,i] in this representation has index
    // k*n_conf_states + i
    std::vector<std::int32_t> i_s;
    std::vector<std::int32_t> j_s;

    // At each markov state i, compute overlap for each combination of two thermodynamic states k and l.
    for (std::size_t i = 0; i < nConfStates; ++i) {
        for (std::size_t k = 0; k < nThermStates; ++k) {
            // therm state must have counts in markov state i
            if (stateCounts.at(k, i) > 0) {
                for (std::size_t l = 0; l < nThermStates; ++l) {
                    // therm state must have counts in markov state i
                    if (k != l && stateCounts.at(l, i) > 0) {
                        if (overlapFunction(i, k, l, ttrajs, dtrajs, biasMatrices, connectivity_factor)) {
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
bool hasOverlapPostHocReplicaExchange(int i, int k, int l, const DTrajs &ttrajs, const DTrajs &dtrajs,
                           const std::vector<np_array_nfc<dtype>> &biasMatrices, dtype connectivity_factor) {
    dtype delta = 0;

    // find all bias energy values for all samples in markov state i sampled at state k and l.
    // store them in two vectors, depending on the thermodynamic state they were sampled at.
    std::vector<std::vector<dtype>> samplesFromK;
    std::vector<std::vector<dtype>> samplesFromL;

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
    dtype n_sum = 0;

    // now compute the overlap between the samples in both vectors
    for (auto el_K : samplesFromK) {
        for (auto el_L: samplesFromL) {
            delta = el_K[0] + el_L[1] - el_K[1] - el_L[0];
            n_sum += std::min(std::exp(delta), 1.0);
        }
    }
    int n = samplesFromK.size();
    int m = samplesFromL.size();

    auto n_avg = n_sum / (n * m);
    return (n + m) * n_avg * connectivity_factor >= 1.0;
}


template<typename dtype>
bool hasOverlapBarVariance(int i, int k, int l, const DTrajs &ttrajs, const DTrajs &dtrajs,
                                      const std::vector<np_array_nfc<dtype>> &biasMatrices, dtype connectivity_factor) {
    return true;
}

template<typename dtype>
extern dtype
_bar_df(np_array_nfc<dtype> db_IJ, std::int32_t L1, np_array_nfc<dtype> db_JI, std::int32_t L2,
        np_array_nfc<dtype> scratch) {
    py::buffer_info db_IJ_buf = db_IJ.request();
    py::buffer_info db_JI_buf = db_JI.request();
    py::buffer_info scratch_buf = scratch.request();

    auto *db_IJ_ptr = (dtype *) db_IJ_buf.ptr;
    auto *db_JI_ptr = (dtype *) db_JI_buf.ptr;
    auto *scratch_ptr = (dtype *) scratch_buf.ptr;

    std::int32_t i;
    dtype ln_avg1;
    dtype ln_avg2;
    for (i = 0; i < L1; i++) {
        scratch_ptr[i] = db_IJ_ptr[i] > 0 ? 0 : db_IJ_ptr[i];
    }
    ln_avg1 = numeric::kahan::logsumexp_sort_kahan_inplace(scratch_ptr, scratch_ptr + L1);
    for (i = 0; i < L1; i++) {
        scratch_ptr[i] = db_JI_ptr[i] > 0 ? 0 : db_JI_ptr[i];
    }
    ln_avg2 = numeric::kahan::logsumexp_sort_kahan_inplace(scratch_ptr, scratch_ptr + L2);
    return ln_avg2 - ln_avg1;
}
}
