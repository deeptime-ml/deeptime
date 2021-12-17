//
// Created by Maaike on 08/12/2021.
//
#pragma once

#if defined(USE_OPENMP)
    #include <omp.h>
#endif

#include <iterator>
#include <algorithm>
#include "tram_types.h"

namespace deeptime::markov::tram {

using IndexList = std::vector<StateIndex>;
using TransitionVector = std::tuple<IndexList, IndexList>;

using Index2D = std::tuple<StateIndex, StateIndex>;
using Indices2D = std::vector<std::vector<Index2D>>;

template<typename dtype>
using BiasPairs = np_array<dtype>;

template<typename dtype>
using PairOfBiasPairs = std::tuple<BiasPairs<dtype>, BiasPairs<dtype>>;


template <typename InIter, typename OutIter>
void flatten (InIter start, InIter end, OutIter dest) {
    while (start != end) {
        dest = std::copy(start->begin(), start->end(), dest);
        ++start;
    }
}

Indices2D findIndexOfSamplesInMarkovState(std::size_t i, const DTrajs *ttrajs, const DTrajs *dtrajs,
                                          StateIndex nThermStates) {
    Indices2D indices(nThermStates);

    for (std::size_t j = 0; j < (*dtrajs).size(); ++j) {
        std::size_t trajLength = (*dtrajs)[j].size();

        auto dtrajBuf = (*dtrajs)[j].template unchecked<1>();
        auto ttrajBuf = ttrajs == nullptr ? nullptr : (*ttrajs)[j].data();

        for (std::size_t n = 0; n < trajLength; ++n) {

            if (dtrajBuf[n] == i) {
                auto k = ttrajs == nullptr ? j : ttrajBuf[n];

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

    static dtype _barDf(const std::vector<dtype> &db_IJ, const std::vector<dtype> &db_JI) {
        std::vector<dtype> scratch(std::max(db_IJ.size(), db_JI.size()), 0);
        std::transform(std::begin(db_IJ), std::end(db_IJ), std::begin(scratch),
                       [](const auto &x) { return std::min(static_cast<dtype>(0), x); });
        auto lnAvg1 = numeric::kahan::logsumexp_sort_kahan_inplace(scratch.begin(), scratch.begin() + db_IJ.size());

        std::transform(std::begin(db_JI), std::end(db_JI), std::begin(scratch),
                       [](const auto &x) { return std::min(static_cast<dtype>(0), x); });
        auto lnAvg2 = numeric::kahan::logsumexp_sort_kahan_inplace(scratch.begin(), scratch.begin() + db_JI.size());

        return lnAvg2 - lnAvg1;
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
        auto df = _barDf(db_IJ, db_JI);

        auto b = std::accumulate(std::begin(du), std::end(du), static_cast<dtype>(0), [df, n, m](auto x, auto y) {
            return x + (1.0 / (2.0 + 2.0 * std::cosh(df - y - std::log(1.0 * static_cast<dtype>(n / m)))));
        });
        return (1 / b - (n + m) / static_cast<dtype>(n * m)) < connectivity_factor;
    }
};

template<typename dtype>
BiasPairs<dtype> getBiasPairs(const std::vector<Index2D> &sampleIndices, StateIndex k, StateIndex l,
                              const BiasMatrices <dtype> *biasMatrices) {
    BiasPairs<dtype> biasPairs (std::vector<std::size_t>{sampleIndices.size(), 2});

    auto biasPairsBuf = biasPairs.template mutable_unchecked<2>();

    for (auto i =0; i < sampleIndices.size(); ++i) {
        auto [j, n] =  sampleIndices[i];
        biasPairsBuf(i, 0) = (*biasMatrices)[j].unchecked()(n, k);
        biasPairsBuf(i, 1) = (*biasMatrices)[j].unchecked()(n, l);
    }
    return biasPairs;
}

template<typename dtype>
PairOfBiasPairs<dtype> getPairOfBiasPairs(const Indices2D &sampleIndicesIn_i, StateIndex k, StateIndex l,
                                          const BiasMatrices <dtype> *biasMatrices) {
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
                                      dtype connectivityFactor,
                                      const py::object *callback = nullptr) {
    // Find all possible transition paths between thermodynamic states. A possible path between thermodynamic states
    // occurs when a transition is possible from [k, i] to [l, i], i.e. we are looking for thermodynamic state pairs
    // that overlap within Markov state i.
    // Whether two thermodynamic states k and l overlap in Markov state i is determined by the samples from k and l that
    // were binned into Markov state i, according to some overlap criterion defined by the overlapFunction and
    // connectivityFactor.

    // i_s and j_s will hold for each markov state all possible transition pairs between thermodynamic states:
    // (i_s[i][n], j_s[i][n]) is one possible transition in markov state i.
    // The therm./Markov state index in unraveled to one dimension, i.e. markov state i in therm state k is represented
    // in these stateVectors as k * nMarkovStates_ + i

    // get the number of threads (if multithreading)
    int nThreads = 1;
    #if defined(USE_OPENMP)
        nThreads = omp_get_num_threads();
    #endif

    // save results in one vector per thread. Concatenate them at the end.
    std::vector<IndexList> thermStateIndicesFromPerThread(nThreads);
    std::vector<IndexList> thermStateIndicesToPerThread(nThreads);

    // threads don't like references
    auto dtrajsPtr = &dtrajs;
    auto ttrajsPtr = ttrajs ? &(*ttrajs) : nullptr;
    auto biasMatricesPtr = &biasMatrices;
    auto stateCountsBuf = stateCounts.template unchecked<2>();

    #pragma omp parallel default(none) firstprivate(nMarkovStates, nThermStates, ttrajsPtr, dtrajsPtr, biasMatricesPtr, stateCountsBuf, connectivityFactor) shared(thermStateIndicesFromPerThread, thermStateIndicesToPerThread)
    {
        IndexList thermStateIndicesFrom;
        IndexList thermStateIndicesTo;

        auto threadNumber = 0;
        #if defined(USE_OPENMP)
            threadNumber = omp_get_thread_num();
        #endif
            std::cout << threadNumber << std::endl;

        // At each markov state i, compute overlap for each combination of two thermodynamic states k and l.
#pragma omp for
        for (StateIndex i = 0; i < nMarkovStates; ++i) {
            // Get all indices in all trajectories of all samples that were binned in markov state i.
            // We use these to determine whether two states overlap in Markov state i.
            Indices2D sampleIndicesIn_i = findIndexOfSamplesInMarkovState(i, ttrajsPtr, dtrajsPtr, nThermStates);

            // Now loop through all thermodynamic state pairs
            for (StateIndex k = 0; k < nThermStates; ++k) {
                // ... states can only overlap if they both have counts in markov state i
                if (stateCountsBuf(k, i) > 0) {
                    for (StateIndex l = 0; l < nThermStates; ++l) {
                        // ... other state also needs to have counts in markov state i.
                        if (k != l && stateCounts.at(l, i) > 0) {
                            // They both have counts! We check if they *really* overlap using the overlap function.
                            // First get all bias energy values that belong to the samples we found.
                            auto biasPairs = getPairOfBiasPairs(sampleIndicesIn_i, k, l, biasMatricesPtr);

                            // Now put the bias values into the overlap function to check if k and l overlap at Markov
                            // state i.
                            if (OverlapMode::hasOverlap(biasPairs, connectivityFactor)) {
                                // They overlap! Save the transition pair by pushing the unraveled indices of both
                                // states to the index lists.
                                auto x = i + k * nMarkovStates;
                                auto y = i + l * nMarkovStates;
                                thermStateIndicesFrom.push_back(x);
                                thermStateIndicesTo.push_back(y);
                            }
                        }
                    }
                }
            }
            // callback for incrementing a progress bar
            if (callback != nullptr) {
                (*callback)();
            }
        }
        // save indices this thread found to the global list of indices
        thermStateIndicesFromPerThread[threadNumber] = thermStateIndicesFrom;
        thermStateIndicesToPerThread[threadNumber] = thermStateIndicesTo;
    }

    // now concatenate all indices to one big list
    IndexList allThermStateIndicesFrom;
    IndexList allThermStateIndicesTo;

    flatten(thermStateIndicesFromPerThread.begin(), thermStateIndicesFromPerThread.end(), allThermStateIndicesFrom.begin());
    flatten(thermStateIndicesToPerThread.begin(), thermStateIndicesToPerThread.end(), allThermStateIndicesTo.begin());

    return std::make_tuple(allThermStateIndicesFrom, allThermStateIndicesTo);
}

}

