//
// Created by marscher on 7/24/17, adapted by clonker.
//


#pragma once

namespace deeptime::clustering::kmeans {

template<typename Metric, typename T>
inline std::tuple<np_array<T>, np_array<int>> cluster(const np_array_nfc<T> &np_chunk,
                                                      const np_array_nfc<T> &np_centers, int n_threads) {
    if (np_chunk.ndim() != 2) {
        throw std::runtime_error(R"(Number of dimensions of "chunk" ain't 2.)");
    }
    if (np_centers.ndim() != 2) {
        throw std::runtime_error(R"(Number of dimensions of "centers" ain't 2.)");
    }

    auto n_frames = np_chunk.shape(0);
    auto dim = np_chunk.shape(1);

    if (dim == 0) {
        throw std::invalid_argument("chunk dimension must be larger than zero.");
    }

    auto chunk = np_chunk.template unchecked<2>();
    auto n_centers = static_cast<std::size_t>(np_centers.shape(0));
    auto centers = np_centers.template unchecked<2>();
    np_array<int> assignments({n_frames});
    auto assignmentsPtr = assignments.mutable_data();

    /* initialize centers_counter and new_centers with zeros */
    std::vector<std::size_t> shape = {n_centers, static_cast<std::size_t>(dim)};
    py::array_t<T> newCenters(shape);
    auto newCentersRef = newCenters.mutable_unchecked();
    std::fill(newCenters.mutable_data(), newCenters.mutable_data() + newCenters.size(), 0.0);
    std::vector<std::size_t> centers_counter(n_centers, 0);

    /* do the clustering */
    if (n_threads == 0) {
        for (py::ssize_t i = 0; i < n_frames; ++i) {
            int argMinDist = 0;
            {
                T minDist = Metric::template compute(&chunk(i, 0), &centers(0, 0), dim);
                for (std::size_t j = 1; j < n_centers; ++j) {
                    auto dist = Metric::template compute(&chunk(i, 0), &centers(j, 0), dim);
                    if (dist < minDist) {
                        minDist = dist;
                        argMinDist = j;
                    }
                }
            }

            {
                assignmentsPtr[i] = argMinDist;
                centers_counter.at(argMinDist)++;
                for (py::ssize_t j = 0; j < dim; j++) {
                    newCentersRef(argMinDist, j) += chunk(i, j);
                }
            }
        }
    } else {
#if defined(USE_OPENMP)
        omp_set_num_threads(n_threads);

#pragma omp parallel for schedule(static, 1)
        for (py::ssize_t i = 0; i < n_frames; ++i) {
            std::vector<T> dists(n_centers);
            for (std::size_t j = 0; j < n_centers; ++j) {
                dists[j] = Metric::template compute(&chunk(i, 0), &centers(j, 0), dim);
            }
#pragma omp flush(dists)

#pragma omp critical(centers_counter)
            {
                auto argMinDist = std::distance(dists.begin(), std::min_element(dists.begin(), dists.end()));
                {
                    assignmentsPtr[i] = argMinDist;
                    centers_counter.at(static_cast<std::size_t>(argMinDist))++;
                    for (py::ssize_t j = 0; j < dim; j++) {
                        newCentersRef(argMinDist, j) += chunk(i, j);
                    }
                }
            }
        }
#else
        {
            std::mutex mutex;

            std::vector<deeptime::thread::scoped_thread> threads;
            threads.reserve(static_cast<std::size_t>(n_threads));

            std::size_t grainSize = n_frames / n_threads;

            auto worker = [&](std::size_t tid, std::size_t begin, std::size_t end, std::mutex& m) {
                for (auto i = begin; i < end; ++i) {
                    std::size_t argMinDist = 0;
                    {
                        T minDist = Metric::template compute(&chunk(i, 0), &centers(0, 0), dim);
                        for (std::size_t j = 1; j < n_centers; ++j) {
                            auto dist = Metric::template compute(&chunk(i, 0), &centers(j, 0), dim);
                            if(dist < minDist) {
                                minDist = dist;
                                argMinDist = j;
                            }
                        }
                    }

                    {
                        std::unique_lock<std::mutex> lock(m);
                        assignmentsPtr[i] = argMinDist;
                        centers_counter.at(argMinDist)++;
                        for (py::ssize_t j = 0; j < dim; j++) {
                            newCentersRef(argMinDist, j) += chunk(i, j);
                        }
                    }
                }
            };

            for(std::uint8_t i = 0; i < n_threads - 1; ++i) {
                threads.emplace_back(worker, i, i*grainSize, (i+1)*grainSize, std::ref(mutex));
            }
            threads.emplace_back(worker, n_threads, (n_threads - 1) * grainSize, n_frames, std::ref(mutex));
        }
#endif
    }

    auto centers_counter_it = centers_counter.begin();
    for (std::size_t i = 0; i < n_centers; ++i, ++centers_counter_it) {
        if (*centers_counter_it == 0) {
            for (py::ssize_t j = 0; j < dim; ++j) {
                newCentersRef(i, j) = centers(i, j);
            }
        } else {
            for (py::ssize_t j = 0; j < dim; ++j) {
                newCentersRef(i, j) /= static_cast<T>(*centers_counter_it);
            }
        }
    }

    return std::make_tuple(newCenters, std::move(assignments));
}

template<typename Metric, typename T>
inline std::tuple<np_array_nfc<T>, int, int, np_array<T>> cluster_loop(
        const np_array_nfc<T> &np_chunk, const np_array_nfc<T> &np_centers,
        int n_threads, int max_iter, T tolerance, py::object &callback) {
    int it = 0;
    bool converged = false;
    T rel_change;
    auto prev_cost = static_cast<T>(0);
    auto currentCenters = np_centers;

    std::vector<T> inertias;
    if (max_iter > 0) {
        inertias.reserve(max_iter);

        do {
            auto clusterResult = cluster<Metric>(np_chunk, currentCenters, n_threads);
            currentCenters = std::get<0>(clusterResult);
            const auto &assignments = std::get<1>(clusterResult);
            auto cost = costFunction<Metric>(np_chunk, currentCenters, assignments, n_threads);
            inertias.push_back(cost);
            rel_change = (cost != 0.0) ? std::abs(cost - prev_cost) / cost : 0;
            prev_cost = cost;
            if (rel_change <= tolerance) {
                converged = true;
            } else {
                if (!callback.is_none()) {
                    /* Acquire GIL before calling Python code */
                    py::gil_scoped_acquire acquire;
                    callback();
                }
            }

            it += 1;
        } while (it < max_iter && !converged);
    }
    int res = max_iter <= 0 || converged ? 0 : 1;
    np_array<T> npInertias({static_cast<py::ssize_t>(inertias.size())});
    std::copy(inertias.begin(), inertias.end(), npInertias.mutable_data());
    return std::make_tuple(currentCenters, res, it, npInertias);
}

template<typename Metric, typename T>
inline T costAssignFunction(const np_array_nfc<T> &np_data, const np_array_nfc<T> &np_centers, int n_threads) {
    auto assignments = assign_chunk_to_centers<Metric>(np_data, np_centers, n_threads);
    return costFunction<Metric>(np_data, np_centers, assignments, n_threads);
}

template<typename Metric, typename T>
inline T costFunction(const np_array_nfc<T> &np_data, const np_array_nfc<T> &np_centers,
                      const np_array<int> &assignments, int n_threads) {
    auto data = np_data.template unchecked<2>();
    auto centers = np_centers.template unchecked<2>();

    T value = static_cast<T>(0);
    auto n_frames = static_cast<std::size_t>(np_data.shape(0));
    auto dim = static_cast<std::size_t>(np_data.shape(1));
    auto assignmentsPtr = assignments.data();
    #ifdef USE_OPENMP
    omp_set_num_threads(n_threads);
    #endif

    #pragma omp parallel for reduction(+:value) default(none) firstprivate(n_frames, data, centers, assignmentsPtr, dim)
    for (std::size_t i = 0; i < n_frames; i++) {
        auto l = Metric::template compute(&data(i, 0), &centers(assignmentsPtr[i], 0), dim);
        {
            value += l * l;
        }
    }
    return value;
}

template<typename Metric, typename dtype>
inline np_array<dtype> initKmeansPlusPlus(const np_array_nfc<dtype> &data, std::size_t k,
                                   std::int64_t seed, int n_threads, py::object &callback) {
    if (static_cast<std::size_t>(data.shape(0)) < k) {
        std::stringstream ss;
        ss << "not enough data to initialize desired number of centers.";
        ss << "Provided frames (" << data.shape(0) << ") < n_centers (" << k << ").";
        throw std::invalid_argument(ss.str());
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
    Distances<dtype> distances = computeDistances<true, Metric>(
            centersPtr, 1, // 1 center picked
            dataPtr, nFrames, // data set
            dim, // dimension
            nullptr, dataNormsSquared.get() // yy precomputed
    );

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
        auto distsToCandidates = computeDistances<true, Metric>(candidatesCoords.data(), nTrials, dataPtr, nFrames, dim,
                                                                nullptr, dataNormsSquared.get());
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

namespace util {
template<typename dtype, typename itype>
inline void assignCenter(itype frameIndex, std::size_t dim, const dtype *const data, dtype *const centers) {
    std::copy(data + frameIndex * dim, data + frameIndex * dim + dim, centers);
}
}

}
