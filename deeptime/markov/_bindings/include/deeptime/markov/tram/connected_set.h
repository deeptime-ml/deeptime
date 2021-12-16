//
// Created by Maaike on 08/12/2021.
//
#pragma once

#include <algorithm>
#include "typedef.h"

namespace deeptime::markov::tram {

using StateVector = std::vector<StateIndex>;
using TransitionVector = std::tuple<StateVector, StateVector>;

using TransitionPair = std::tuple<StateIndex, StateIndex>;
using IndexList = std::vector<std::vector<TransitionPair>>;

template<typename dtype>
using BiasPairs = np_array<dtype>;

template<typename dtype>
using PairOfBiasPairs = std::tuple<BiasPairs<dtype>, BiasPairs<dtype>>;


IndexList findIndexOfSamplesInMarkovState(std::size_t i, const std::optional<DTrajs> &ttrajs, const DTrajs &dtrajs,
                                          StateIndex nThermStates) {
    IndexList indices(nThermStates);

    for (std::size_t j = 0; j < dtrajs.size(); ++j) {
        std::size_t trajLength = dtrajs[j].size();

        auto dtrajBuf = dtrajs[j].template unchecked<1>();
        auto ttrajBuf = ttrajs ? (*ttrajs)[j].data() : nullptr;

        for (std::size_t n = 0; n < trajLength; ++n) {

            if (dtrajBuf[n] == i) {
                auto k = ttrajs ? ttrajBuf[n] : j;

                // markov state i sampled in therm state k can be found at bias matrix index (j, n,)
                indices[k].emplace_back(j, n);
            }
        }
    }
    return indices;
}


template<typename dtype>
struct OverlapPostHocReplicaExchange {
    static bool hasOverlap(const PairOfBiasPairs<dtype> &biasPairs, dtype connectivity_factor) {
        auto &[biasesSampledAtK, biasesSampledAtL] = biasPairs;
        auto biasPairsBufK = biasesSampledAtK.template unchecked<2>();
        auto biasPairsBufL = biasesSampledAtL.template unchecked<2>();
        auto n = biasesSampledAtK.shape(0);
        auto m = biasesSampledAtL.shape(0);

        dtype n_sum = 0;

        // now compute the overlap between the samples in both vectors
        for (auto i = 0; i < n; ++i) {
            for (auto j = 0; j < m; ++j) {
                dtype delta = biasPairsBufK(i,0) + biasPairsBufL(j,1) - biasPairsBufK(i,1) - biasPairsBufL(j,0);
                n_sum += std::min(std::exp(delta), 1.0);
            }
        }

        dtype n_avg = n_sum / (n * m);
        return (n + m) * n_avg * connectivity_factor >= 1.0;
    }
};


template<typename dtype>
struct OverlapBarVariance {

    static dtype
    _bar_df(const std::vector<dtype> &db_IJ, std::size_t L1, const std::vector<dtype> &db_JI, std::size_t L2) {
        std::vector<dtype> scratch(L1 + L2, 0);

        dtype ln_avg1;
        dtype ln_avg2;
        for (std::size_t i = 0; i < L1; i++) {
            scratch[i] = db_IJ[i] > 0 ? 0 : db_IJ[i];
        }
        ln_avg1 = numeric::kahan::logsumexp_sort_kahan_inplace(scratch.begin(), L1);
        for (std::size_t i = 0; i < L2; i++) {
            scratch[i] = db_JI[i] > 0 ? 0 : db_JI[i];
        }
        ln_avg2 = numeric::kahan::logsumexp_sort_kahan_inplace(scratch.begin(), L2);
        return ln_avg2 - ln_avg1;
    }

    static bool hasOverlap(const PairOfBiasPairs<dtype> &biasPairs, dtype connectivity_factor) {
        auto &[biasesSampledAtK, biasesSampledAtL] = biasPairs;
        auto biasPairsBufK = biasesSampledAtK.template unchecked<2>();
        auto biasPairsBufL = biasesSampledAtL.template unchecked<2>();

        auto n = biasesSampledAtK.shape(0);
        auto m = biasesSampledAtL.shape(0);

        std::vector<dtype> db_IJ(n);
        std::vector<dtype> db_JI(m);
        std::vector<dtype> du(n + m);

        for (std::size_t i = 0; i < n; ++i) {
            db_IJ[i] = biasPairsBufK(i, 1) - biasPairsBufK(i, 0);
            du[i] = db_IJ[i];
        }
        for (std::size_t i = 0; i < m; ++i) {
            db_JI[i] = biasPairsBufL(i, 0) - biasPairsBufL(i, 1);
            du[n + i] = -db_JI[i];
        }
        auto df = _bar_df(db_IJ, n, db_JI, m);

        dtype b = 0;
        for (std::size_t i = 0; i < n + m; ++i) {
            b += (1.0 / (2.0 + 2.0 * std::cosh(df - du[i] - std::log(1.0 * static_cast<dtype>(n / m)))));
        }
        return (1 / b - (n + m) / static_cast<dtype>(n * m)) < connectivity_factor;
    }
};

template<typename dtype>
BiasPairs<dtype> getBiasPairs(const std::vector<TransitionPair> &sampleIndices, StateIndex k, StateIndex l,
                              const BiasMatrices <dtype> &biasMatrices) {
    BiasPairs<dtype> biasPairs (std::vector<std::size_t>{sampleIndices.size(), 2});

    auto biasPairsBuf = biasPairs.template mutable_unchecked<2>();

    for (auto i =0; i < sampleIndices.size(); ++i) {
        auto [j, n] =  sampleIndices[i];
        biasPairsBuf(i, 0) = biasMatrices[j].unchecked()(n, k);
        biasPairsBuf(i, 1) = biasMatrices[j].unchecked()(n, l);
    }
    return biasPairs;
}

template<typename dtype>
PairOfBiasPairs<dtype> getPairOfBiasPairs(const IndexList &sampleIndicesIn_i, StateIndex k, StateIndex l,
                                                const BiasMatrices <dtype> &biasMatrices) {
    auto biasPairsK = getBiasPairs(sampleIndicesIn_i[k], k, l, biasMatrices);
    auto biasPairsL = getBiasPairs(sampleIndicesIn_i[l], k, l, biasMatrices);

    return std::make_tuple(biasPairsK, biasPairsL);
}

template<typename dtype, typename OverlapMode>
TransitionVector findStateTransitions(const std::optional<DTrajs> &ttrajs,
                                      const DTrajs &dtrajs,
                                      const BiasMatrices <dtype> &biasMatrices,
                                      const np_array <std::int32_t> &stateCounts,
                                      StateIndex nThermStates,
                                      StateIndex nMarkovStates,
                                      dtype connectivityFactor) {
    // Find all possible transition paths between thermodynamic states. A possible path between thermodynamic states
    // occurs when a transition is possible from [k, i] to [l, i], i.e. we are looking for thermodynamic state pairs
    // that overlap within Markov state i.
    // Whether two thermodynamic states k and l overlap in Markov state i is determined by the samples from k and l that
    // were binned into Markov state i, according to some overlap criterion defined by the overlapFunction and
    // connectivityFactor.

    std::cout << "Finding connected sets..." << std::endl;

    // i_s and j_s will hold all possible transition pairs: (i_s[n], j_s[n]) is one possible transition.
    // The therm./Markov state index in unraveled to one dimension, i.e. markov state i in therm state k is represented
    // in these stateVectors as k * nMarkovStates_ + i
    StateVector i_s;
    StateVector j_s;

    // todo can this be parallelized?
    // At each markov state i, compute overlap for each combination of two thermodynamic states k and l.
    //  #pragma omp parallel for default(none) firstprivate(i_s, j_s, nMarkovStates, nThermStates, ttrajs, dtrajs, biasMatrices, stateCounts, connectivityFactor)
    for (StateIndex i = 0; i < nMarkovStates; ++i) {
        std::cout << "Markov state: " << i << std::endl;

        // Get all indices in all trajectories of all samples that were binned in markov state i.
        IndexList sampleIndicesIn_i = findIndexOfSamplesInMarkovState(i, ttrajs, dtrajs, nThermStates);

        for (StateIndex k = 0; k < nThermStates; ++k) {
            std::cout << "Find overlap of all states with therm. state: " << k << std::endl;
            // therm state must have counts in markov state i
            if (stateCounts.at(k, i) > 0) {
                for (StateIndex l = 0; l < nThermStates; ++l) {
                    // therm state must have counts in markov state i
                    if (k != l && stateCounts.at(l, i) > 0) {
                        auto biasPairs = getPairOfBiasPairs(sampleIndicesIn_i, k, l, biasMatrices);

                        // check if states k and l overlap at Markov state i.
                        if (OverlapMode::hasOverlap(biasPairs, connectivityFactor)) {
                            // push the unraveled index of the therm state transition to the transition list.
                            auto x = i + k * nMarkovStates;
                            auto y = i + l * nMarkovStates;
                            i_s.push_back(x);
                            j_s.push_back(y);
                        }
                    }
                }
            }
        }
    }
    return std::make_tuple(i_s, j_s);
}

}

