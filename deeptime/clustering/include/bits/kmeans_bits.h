//
// Created by marscher on 7/24/17, adapted by clonker.
//


#pragma once

#include "kmeans.h"

namespace deeptime {
namespace clustering {
namespace kmeans {

template<typename T>
inline std::tuple<np_array<T>, np_array<int>> cluster(const np_array_nfc<T> &np_chunk,
                                                      const np_array_nfc<T> &np_centers, int n_threads,
                                                      const Metric *metric) {
    if (metric == nullptr) {
        metric = default_metric();
    }

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
        for (pybind11::ssize_t i = 0; i < n_frames; ++i) {
            int argMinDist = 0;
            {
                T minDist = metric->compute(&chunk(i, 0), &centers(0, 0), dim);
                for (std::size_t j = 1; j < n_centers; ++j) {
                    auto dist = metric->compute(&chunk(i, 0), &centers(j, 0), dim);
                    if (dist < minDist) {
                        minDist = dist;
                        argMinDist = j;
                    }
                }
            }

            {
                assignmentsPtr[i] = argMinDist;
                centers_counter.at(argMinDist)++;
                for (pybind11::ssize_t j = 0; j < dim; j++) {
                    newCentersRef(argMinDist, j) += chunk(i, j);
                }
            }
        }
    } else {
#if defined(USE_OPENMP)
        omp_set_num_threads(n_threads);

#pragma omp parallel for schedule(static, 1)
        for (pybind11::ssize_t i = 0; i < n_frames; ++i) {
            std::vector<T> dists(n_centers);
            for (std::size_t j = 0; j < n_centers; ++j) {
                dists[j] = metric->compute(&chunk(i, 0), &centers(j, 0), dim);
            }
#pragma omp flush(dists)

#pragma omp critical(centers_counter)
            {
                auto argMinDist = std::distance(dists.begin(), std::min_element(dists.begin(), dists.end()));
                {
                    assignmentsPtr[i] = argMinDist;
                    centers_counter.at(static_cast<std::size_t>(argMinDist))++;
                    for (pybind11::ssize_t j = 0; j < dim; j++) {
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
                        T minDist = metric->compute(&chunk(i, 0), &centers(0, 0), dim);
                        for (std::size_t j = 1; j < n_centers; ++j) {
                            auto dist = metric->compute(&chunk(i, 0), &centers(j, 0), dim);
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
                        for (pybind11::ssize_t j = 0; j < dim; j++) {
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
            for (pybind11::ssize_t j = 0; j < dim; ++j) {
                newCentersRef(i, j) = centers(i, j);
            }
        } else {
            for (pybind11::ssize_t j = 0; j < dim; ++j) {
                newCentersRef(i, j) /= static_cast<T>(*centers_counter_it);
            }
        }
    }

    return std::make_tuple(newCenters, std::move(assignments));
}

template<typename T>
inline std::tuple<np_array_nfc<T>, int, int, np_array<T>> cluster_loop(
        const np_array_nfc<T> &np_chunk, const np_array_nfc<T> &np_centers,
        int n_threads, int max_iter, T tolerance, py::object &callback, const Metric *metric) {
    if (metric == nullptr) {
        metric = default_metric();
    }
    int it = 0;
    bool converged = false;
    T rel_change;
    auto prev_cost = static_cast<T>(0);
    auto currentCenters = np_centers;

    std::vector<T> inertias;
    inertias.reserve(max_iter);

    do {
        auto clusterResult = cluster<T>(np_chunk, currentCenters, n_threads, metric);
        currentCenters = std::get<0>(clusterResult);
        const auto &assignments = std::get<1>(clusterResult);
        auto cost = costFunction(np_chunk, currentCenters, assignments, n_threads, metric);
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
    int res = converged ? 0 : 1;
    np_array<T> npInertias({static_cast<pybind11::ssize_t>(inertias.size())});
    std::copy(inertias.begin(), inertias.end(), npInertias.mutable_data());
    return std::make_tuple(currentCenters, res, it, npInertias);
}

template<typename T>
inline T costFunction(const np_array_nfc<T> &np_data, const np_array_nfc<T> &np_centers,
                      const np_array<int> &assignments, int n_threads, const Metric *metric) {
    if(metric == nullptr) {
        metric = default_metric();
    }
    auto data = np_data.template unchecked<2>();
    auto centers = np_centers.template unchecked<2>();

    T value = static_cast<T>(0);
    auto n_frames = static_cast<std::size_t>(np_data.shape(0));
    auto dim = static_cast<std::size_t>(np_data.shape(1));
    auto assignmentsPtr = assignments.data();
    #ifdef USE_OPENMP
    omp_set_num_threads(n_threads);
    #endif

    #pragma omp parallel for reduction(+:value) default(none) firstprivate(n_frames, metric, data, centers, assignmentsPtr, dim)
    for (std::size_t i = 0; i < n_frames; i++) {
        auto l = metric->compute(&data(i, 0), &centers(assignmentsPtr[i], 0), dim);
        {
            value += l * l;
        }
    }
    return value;
}

}
}
}
