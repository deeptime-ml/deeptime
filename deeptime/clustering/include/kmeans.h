//
// Created by marscher on 4/3/17, adapted by clonker.
//

#pragma once

#include <utility>
#include <random>
#include <atomic>

#include <pybind11/pytypes.h>
#include <mutex>

#include "common.h"
#include "metric.h"
#include "thread_utils.h"
#include "distribution_utils.h"

namespace deeptime {
namespace clustering {
namespace kmeans {

template<typename T>
std::tuple<np_array<T>, np_array<int>> cluster(const np_array_nfc<T>& np_chunk, const np_array_nfc<T>& np_centers,
                                               int n_threads, const Metric *metric);

template<typename T>
std::tuple<np_array_nfc<T>, int, int, np_array<T>> cluster_loop(
        const np_array_nfc<T> &np_chunk, const np_array_nfc<T> &np_centers,
        int n_threads, int max_iter, T tolerance, py::object &callback, const Metric *metric);


template<typename T>
T costFunction(const np_array_nfc<T> &np_data, const np_array_nfc<T> &np_centers,
               const np_array<int> &assignments, int n_threads, const Metric *metric);

template<typename T>
T costAssignFunction(const np_array_nfc<T> &np_data, const np_array_nfc<T> &np_centers,
               int n_threads, const Metric *metric) {
    auto assignments = assign_chunk_to_centers(np_data, np_centers, n_threads, metric);
    return costFunction(np_data, np_centers, assignments, n_threads, metric);
}

namespace util {
template<typename dtype, typename itype>
void assignCenter(itype frameIndex, std::size_t dim, const dtype *const data, dtype *const centers) {
    std::copy(data + frameIndex * dim, data + frameIndex * dim + dim, centers);
}
}

template<typename dtype>
np_array<dtype> initKmeansPlusPlus(const np_array_nfc<dtype> &data, std::size_t k,
                                   std::int64_t seed, int n_threads, py::object &callback, const Metric *metric) {
    if (static_cast<std::size_t>(data.shape(0)) < k) {
        std::stringstream ss;
        ss << "not enough data to initialize desired number of centers.";
        ss << "Provided frames (" << data.shape(0) << ") < n_centers (" << k << ").";
        throw std::invalid_argument(ss.str());
    }

    if(metric == nullptr) {
        metric = default_metric();
    }

    #ifdef USE_OPENMP
    omp_set_num_threads(n_threads);
    #endif

    if (data.ndim() != 2) {
        throw std::invalid_argument("input data does not have two dimensions.");
    }

    auto dim = static_cast<std::size_t>(data.shape(1));
    auto nFrames = static_cast<std::size_t>(data.shape(0));

    // random generator
    auto generator = seed < 0 ? rnd::randomlySeededGenerator() : rnd::seededGenerator(seed);
    std::uniform_int_distribution<std::int64_t> uniform(0, nFrames - 1);
    std::uniform_real_distribution<double> uniformReal(0, 1);

    // number of trials before choosing the data point with the best potential
    auto nTrials = static_cast<std::size_t>(2 + std::log(k));

    np_array<dtype> centers({k, dim});

    const dtype* dataPtr = data.data();
    dtype* centersPtr = centers.mutable_data();

    // precompute xx
    auto dataNormsSquared = precomputeXX(dataPtr, nFrames, dim);

    {
        // select first center random uniform
        auto firstCenterIx = uniform(generator);
        // copy first center into centers array
        util::assignCenter(firstCenterIx, dim, dataPtr, centersPtr);
        // perform callback
        if (!callback.is_none()) {
            py::gil_scoped_acquire acquire;
            callback();
        }
    }

    // 1 x nFrames distance matrix
    Distances<dtype> distances = computeDistances<true>(
            centersPtr, 1, // 1 center picked
            dataPtr, nFrames, // data set
            dim, // dimension
            nullptr, dataNormsSquared.get(), // yy precomputed
            metric);

    double currentPotential {0};
    std::vector<dtype> distancesCumsum (distances.size(), 0);

    {
        // compute cumulative sum of distances and sum over all distances as last element of cumsum
        std::partial_sum(distances.begin(), distances.end(), distancesCumsum.begin());
        currentPotential = distancesCumsum.back();
    }

    auto trialGenerator = [&generator, uniformReal, &currentPotential]() mutable {
        return currentPotential * uniformReal(generator);
    };

    std::vector<double> trialValues (nTrials, 0);
    std::vector<std::size_t> candidatesIds (nTrials, 0);
    std::vector<double> candidatesPotentials (nTrials, 0);
    std::vector<dtype> candidatesCoords (nTrials * dim, 0);
    for(std::size_t c = 1; c < k; ++c) {
        // fill trial values with potential-weighted uniform dist.
        std::generate(std::begin(trialValues), std::end(trialValues), trialGenerator);
        std::sort(std::begin(trialValues), std::end(trialValues));

        // look for trial values in cumsum of distances
        {
            // since trial values are sorted we can make binary search on cumsum and take lower bound of previous
            // search as begin of next search
            auto currentLowerIt = distancesCumsum.begin();
            for(std::size_t i = 0; i < trialValues.size(); ++i) {
                currentLowerIt = std::lower_bound(currentLowerIt, distancesCumsum.end(), trialValues[i]);
                if (currentLowerIt != distancesCumsum.end()) {
                    candidatesIds[i] = static_cast<std::size_t>(std::distance(distancesCumsum.begin(), currentLowerIt));
                } else {
                    candidatesIds[i] = nFrames - 1;
                }
                // copy frame to candidates coords storage
                std::copy(dataPtr + candidatesIds[i] * dim, dataPtr + candidatesIds[i] * dim + dim,
                          candidatesCoords.begin() + i*dim);
            }
        }
        // nTrials x nFrames distance matrix
        auto distsToCandidates = computeDistances<true>(candidatesCoords.data(), nTrials, dataPtr, nFrames, dim,
                                                        nullptr, dataNormsSquared.get(), metric);
        // update with current best distances
        auto distsToCandidatesPtr = distsToCandidates.data();
        auto distancesPtr = distances.data();
        #pragma omp parallel for collapse(2) default(none) firstprivate(nTrials, nFrames, distsToCandidatesPtr, distancesPtr)
        for(std::size_t trial = 0; trial < nTrials; ++trial) {
            for(std::size_t frame = 0; frame < nFrames; ++frame) {
                dtype dist = distsToCandidatesPtr[trial*nFrames + frame];
                distsToCandidatesPtr[trial*nFrames + frame] = std::min(dist, distancesPtr[frame]);
            }
        }

        // compute potentials for trials
        {
            for (std::size_t trial = 0; trial < nTrials; ++trial) {
                auto* dptr = distsToCandidates.data() + trial * nFrames;

                dtype trialPotential{0};
                #pragma omp parallel for reduction(+:trialPotential) default(none) firstprivate(nFrames, dptr)
                for (std::size_t t = 0; t < nFrames; ++t) {
                    trialPotential += dptr[t];
                }

                candidatesPotentials[trial] = trialPotential;
            }
        }

        // best candidate
        auto argminIt = std::min_element(candidatesPotentials.begin(), candidatesPotentials.end());
        auto bestCandidateIx = std::distance(candidatesPotentials.begin(), argminIt);
        auto bestCandidateId = candidatesIds[bestCandidateIx];

        // update with best candidate and repeat or stop
        currentPotential = candidatesPotentials[bestCandidateIx];
        // update distances to last picked center
        std::copy(distsToCandidates.data() + bestCandidateIx * nFrames, distsToCandidates.data() + bestCandidateIx*nFrames + nFrames, distances.data());
        // update cumsum
        std::partial_sum(distances.begin(), distances.end(), distancesCumsum.begin());
        // set center
        util::assignCenter(bestCandidateId, dim, dataPtr, centersPtr + c * dim);

        // perform callback
        if (!callback.is_none()) {
            py::gil_scoped_acquire acquire;
            callback();
        }
    }

    return centers;
}

}

}
}
#include "bits/kmeans_bits.h"
