//
// Created by Maaike on 08/12/2021.
//
#pragma once

#if defined(USE_OPENMP)

#include <omp.h>

#endif

#include <iterator>
#include <algorithm>
#include "common.h"

namespace deeptime::markov::tram {

using IndexList = std::vector<StateIndex>;
using TransitionVector = std::tuple<IndexList, IndexList>;

using Index2D = std::tuple<StateIndex, StateIndex>;
using Indices2D = std::vector<std::vector<Index2D>>;

template<typename dtype>
using BiasPair = std::tuple<dtype, dtype>;

template<typename dtype>
using BiasPairs = std::vector<BiasPair<dtype>>;

template<typename dtype>
using PairOfBiasPairs = std::tuple<BiasPairs<dtype>, BiasPairs<dtype>>;


template<typename InIter, typename OutIter>
void flatten(InIter start, InIter end, OutIter dest) {
    while (start != end) {
        dest = std::copy(start->begin(), start->end(), dest);
        ++start;
    }
}

template<typename DTrajs>
Indices2D findIndexOfSamplesInMarkovState(StateIndex i, std::size_t nTrajs, const DTrajs *ttrajs, const DTrajs *dtrajs,
                                          StateIndex nThermStates) {
    Indices2D indices(nThermStates);

    for (std::size_t j = 0; j < nTrajs; ++j) {
        std::size_t trajLength = dtrajs[j].size();

        for (std::size_t n = 0; n < trajLength; ++n) {
            // cast i because dtraj can contain negative state indices
            if (dtrajs[j](n) == static_cast<std::int32_t>(i)) {
                auto k = ttrajs ? ttrajs[j](n) : j;

                // markov state i sampled in therm state k can be found at bias matrix index (j, n,)
                indices[k].emplace_back(j, n);
            }
        }
    }
    return indices;
}


template<typename dtype>
struct OverlapPostHocReplicaExchange {
    static bool hasOverlap(const PairOfBiasPairs<dtype> &biasPairs, dtype connectivityFactor) {
        auto &[biasesSampledAtK, biasesSampledAtL] = biasPairs;
        auto n = biasesSampledAtK.size();
        auto m = biasesSampledAtL.size();

        dtype n_sum = 0;

        // now compute the overlap between the samples in both vectors
        for (auto[a_k, a_l]: biasesSampledAtK) {
            for (auto[b_k, b_l]: biasesSampledAtL) {
                dtype delta = a_k + b_l - a_l - b_k;
                n_sum += std::min(std::exp(delta), 1.0);
            }
        }

        dtype n_avg = n_sum / (n * m);
        return (n + m) * n_avg * connectivityFactor >= 1.0;
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
        const auto &[biasesSampledAtK, biasesSampledAtL] = biasPairs;

        auto n = biasesSampledAtK.size();
        auto m = biasesSampledAtL.size();

        std::vector<dtype> db_IJ(n);
        std::vector<dtype> db_JI(m);
        std::vector<dtype> du(n + m);

        for (decltype(n) i = 0; i < n; ++i) {
            auto[a_k, a_l] = biasesSampledAtK[i];
            db_IJ[i] = a_l - a_k;
            du[i] = db_IJ[i];
        }
        for (decltype(m) i = 0; i < m; ++i) {
            auto[b_k, b_l] = biasesSampledAtL[i];
            db_JI[i] = b_k - b_l;
            du[n + i] = -db_JI[i];
        }
        auto df = _barDf(db_IJ, db_JI);

        auto b = std::accumulate(std::begin(du), std::end(du), static_cast<dtype>(0), [df, n, m](auto x, auto y) {
            return x + (1.0 / (2.0 + 2.0 * std::cosh(df - y - std::log(1.0 * static_cast<dtype>(n / m)))));
        });
        return (1 / b - (n + m) / static_cast<dtype>(n * m)) < connectivity_factor;
    }
};

template<typename BiasMatrices>
const auto getBiasPairs(const std::vector<Index2D> &sampleIndices, StateIndex k, StateIndex l,
                        const BiasMatrices *biasMatrices) {

    using dtype = typename BiasMatrices::value_type;

    BiasPairs<dtype> biasPairs;
    biasPairs.reserve(sampleIndices.size());

    for (std::size_t i = 0; i < sampleIndices.size(); ++i) {
        auto[j, n] =  sampleIndices[i];
        biasPairs.emplace_back(biasMatrices[j](n, k), biasMatrices[j](n, l));
    }
    return biasPairs;
}

template<typename BiasMatrices>
const auto getPairOfBiasPairs(const Indices2D &sampleIndicesIn_i, StateIndex k, StateIndex l,
                              const BiasMatrices *biasMatrices) {

    auto biasPairsK = getBiasPairs(sampleIndicesIn_i[k], k, l, biasMatrices);
    auto biasPairsL = getBiasPairs(sampleIndicesIn_i[l], k, l, biasMatrices);

    return std::make_tuple(std::move(biasPairsK), std::move(biasPairsL));
}

template<typename dtype, typename OverlapMode>
TransitionVector findStateTransitions(const std::optional<DTrajs> &ttrajs,
                                      const DTrajs &dtrajs,
                                      const BiasMatrices <dtype> &biasMatrices,
                                      const CountsMatrix &stateCounts,
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
#pragma omp parallel default(none) shared(nThreads)
    {
#pragma omp master
        nThreads = omp_get_num_threads();
    }
#endif

    // save results in one vector per thread. Concatenate them at the end.
    std::vector<IndexList> thermStateIndicesFromPerThread(nThreads);
    std::vector<IndexList> thermStateIndicesToPerThread(nThreads);

    auto nTrajs = dtrajs.size();

    // threads don't like references
    std::vector<ArrayBuffer<DTraj, 1>> dtrajBuffers(dtrajs.begin(), dtrajs.end());
    auto dtrajsPtr = dtrajBuffers.data();

    std::vector<ArrayBuffer<DTraj, 1>> ttrajBuffers;
    if (ttrajs) {
        ttrajBuffers = std::vector<ArrayBuffer<DTraj, 1>>((*ttrajs).begin(), (*ttrajs).end());
    }
    const auto* ttrajsPtr = ttrajBuffers.data();

    std::vector<ArrayBuffer<BiasMatrix<dtype>, 2>> biasMatrixBuffers(biasMatrices.begin(), biasMatrices.end());
    auto biasMatricesPtr = biasMatrixBuffers.data();

    ArrayBuffer<CountsMatrix, 2> stateCountsBuf{stateCounts};
    auto stateCountsBufPtr = &stateCountsBuf;

    #pragma omp parallel default(none) firstprivate(nMarkovStates, nThermStates, nTrajs, ttrajsPtr, dtrajsPtr, \
    biasMatricesPtr, stateCountsBufPtr, connectivityFactor, callback) shared(thermStateIndicesFromPerThread, \
    thermStateIndicesToPerThread)
    {
        IndexList thermStateIndicesFrom;
        IndexList thermStateIndicesTo;

        int threadNumber = 0;
        #if defined(USE_OPENMP)
            threadNumber = omp_get_thread_num();
        #endif

        // At each markov state i, compute overlap for each combination of two thermodynamic states k and l.
        #pragma omp for
        for (StateIndex i = 0; i < nMarkovStates; ++i) {

            // Get all indices in all trajectories of all samples that were binned in markov state i.
            // We use these to determine whether two states overlap in Markov state i.
            Indices2D sampleIndicesIn_i = findIndexOfSamplesInMarkovState(i, nTrajs, ttrajsPtr, dtrajsPtr,
                                                                          nThermStates);

            // Now loop through all thermodynamic state pairs
            for (StateIndex k = 0; k < nThermStates; ++k) {
                // callback for incrementing a progress bar
                if (callback != nullptr) {
                    #pragma omp critical
                    {
                        py::gil_scoped_acquire guard;
                        (*callback)();
                    }
                }
                // ... states can only overlap if they both have counts in markov state i
                if ((*stateCountsBufPtr)(k, i) > 0) {
                    for (StateIndex l = 0; l < nThermStates; ++l) {
                        // ... other state also needs to have counts in markov state i.
                        if (k != l && (*stateCountsBufPtr)(l, i) > 0) {
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
        }
        // save indices this thread found to the global list of indices
        thermStateIndicesFromPerThread[threadNumber] = thermStateIndicesFrom;
        thermStateIndicesToPerThread[threadNumber] = thermStateIndicesTo;
    }

    // now concatenate all indices to one big list
    IndexList allThermStateIndicesFrom;
    IndexList allThermStateIndicesTo;

    flatten(thermStateIndicesFromPerThread.begin(), thermStateIndicesFromPerThread.end(),
            std::back_inserter(allThermStateIndicesFrom));
    flatten(thermStateIndicesToPerThread.begin(), thermStateIndicesToPerThread.end(),
            std::back_inserter(allThermStateIndicesTo));

    return std::make_tuple(allThermStateIndicesFrom, allThermStateIndicesTo);
}

}
