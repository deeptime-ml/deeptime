//
// Created by marscher on 7/24/17.
//


#ifndef PYEMMA_KMEANS_BITS_H_H
#define PYEMMA_KMEANS_BITS_H_H

#include "kmeans.h"
#include "threading_utils.h"

#include <random>
#include <atomic>

#include <pybind11/pytypes.h>
#include <mutex>


template<typename dtype>
typename KMeans<dtype>::np_array
KMeans<dtype>::cluster(const np_array &np_chunk, const np_array &np_centers, int n_threads) const {

    if (np_chunk.ndim() != 2) {
        throw std::runtime_error(R"(Number of dimensions of "chunk" ain't 2.)");
    }
    if (np_centers.ndim() != 2) {
        throw std::runtime_error(R"(Number of dimensions of "centers" ain't 2.)");
    }

    auto n_frames = static_cast<size_t>(np_chunk.shape(0));
    auto dim = static_cast<size_t>(np_chunk.shape(1));

    if (dim == 0) {
        throw std::invalid_argument("chunk dimension must be larger than zero.");
    }

    auto chunk = np_chunk.template unchecked<2>();
    auto n_centers = static_cast<size_t>(np_centers.shape(0));
    auto centers = np_centers.template unchecked<2>();

    /* initialize centers_counter and new_centers with zeros */
    std::vector<std::size_t> shape = {n_centers, dim};
    py::array_t <dtype> return_new_centers(shape);
    auto new_centers = return_new_centers.mutable_unchecked();
    std::fill(return_new_centers.mutable_data(), return_new_centers.mutable_data() + return_new_centers.size(), 0.0);
    std::vector<std::size_t> centers_counter(n_centers, 0);

    /* do the clustering */
    if (n_threads == 0) {
        std::size_t closest_center_index = 0;
        for (std::size_t i = 0; i < n_frames; ++i) {
            auto mindist = std::numeric_limits<dtype>::max();
            for(std::size_t j = 0; j < n_centers; ++j) {
                auto d = parent_t::metric->compute(&chunk(i, 0), &centers(j, 0));
                if(d < mindist) {
                    mindist = d;
                    closest_center_index = j;
                }
            }
            centers_counter[closest_center_index]++;
            for (std::size_t j = 0; j < dim; j++) {
                new_centers(closest_center_index, j) += chunk(i, j);
            }
        }
    } else {
#if defined(USE_OPENMP)
        omp_set_num_threads(n_threads);

#pragma omp parallel for schedule(static, 1)
        for (std::size_t i = 0; i < n_frames; ++i) {
            std::vector<dtype> dists(n_centers);
            for (std::size_t j = 0; j < n_centers; ++j) {
                dists[j] = parent_t::metric->compute(&chunk(i, 0), &centers(j, 0));
            }
#pragma omp flush(dists)

#pragma omp critical(centers_counter)
            {
                auto closest_center_index = std::distance(dists.begin(), std::min_element(dists.begin(), dists.end()));
                {
                    centers_counter.at(static_cast<std::size_t>(closest_center_index))++;
                    for (std::size_t j = 0; j < dim; j++) {
                        new_centers(closest_center_index, j) += chunk(i, j);
                    }
                }
            }
        }
#else
        {
            std::mutex mutex;

            std::vector<scoped_thread> threads;
            threads.reserve(static_cast<std::size_t>(n_threads));

            std::size_t grainSize = n_frames / n_threads;

            auto worker = [&](std::size_t tid, std::size_t begin, std::size_t end, std::mutex& m) {
                for (auto i = begin; i < end; ++i) {
                    std::size_t argMinDist = 0;
                    {
                        dtype minDist = parent_t::metric->compute(&chunk(i, 0), &centers(0, 0));
                        for (std::size_t j = 1; j < n_centers; ++j) {
                            auto dist = parent_t::metric->compute(&chunk(i, 0), &centers(j, 0));
                            if(dist < minDist) {
                                minDist = dist;
                                argMinDist = j;
                            }
                        }
                    }

                    {
                        std::unique_lock<std::mutex> lock(m);

                        centers_counter.at(argMinDist)++;
                        for (std::size_t j = 0; j < dim; j++) {
                            new_centers(argMinDist, j) += chunk(i, j);
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
            for (std::size_t j = 0; j < dim; ++j) {
                new_centers(i, j) = centers(i, j);
            }
        } else {
            for (std::size_t j = 0; j < dim; ++j) {
                new_centers(i, j) /= static_cast<dtype>(*centers_counter_it);
            }
        }
    }
    return return_new_centers;
}


template<typename dtype>
typename KMeans<dtype>::cluster_res KMeans<dtype>::cluster_loop(const np_array& np_chunk, np_array& np_centers,
                                                                int n_threads, int max_iter, float tolerance,
                                                                py::object& callback) const {
    int it = 0;
    bool converged = false;
    dtype rel_change = std::numeric_limits<dtype>::max();
    dtype prev_cost = 0;
    do {
        np_centers = cluster(np_chunk, np_centers, n_threads);
        auto cost = costFunction(np_chunk, np_centers, n_threads);
        rel_change = (cost != 0.0) ? std::abs(cost - prev_cost) / cost : 0;
        prev_cost = cost;
        if(rel_change <= tolerance) {
            converged = true;
        } else {
            if(! callback.is_none()) {
                /* Acquire GIL before calling Python code */
                py::gil_scoped_acquire acquire;
                callback();
            }
        }

        it += 1;
    } while(it < max_iter && ! converged);
    int res = converged ? 0 : 1;
    return std::make_tuple(std::move(np_centers), res, it);
}

template<typename dtype>
dtype KMeans<dtype>::costFunction(const np_array &np_data, const np_array &np_centers, int n_threads) const {
    auto data = np_data.template unchecked<2>();
    auto centers = np_centers.template unchecked<2>();

    dtype value = 0.0;
    auto n_frames = static_cast<std::size_t>(np_data.shape(0));
#ifdef USE_OPENMP
    omp_set_num_threads(n_threads);
#endif

#pragma omp parallel for reduction(+:value)
    for (std::size_t i = 0; i < n_frames; i++) {
        for (std::size_t r = 0; r < np_centers.shape(0); r++) {
            auto l = parent_t::metric->compute(&data(i, 0), &centers(r, 0));
            {
                value += l;
            }
        }
    }
    return value;
}

template<typename dtype>
typename KMeans<dtype>::np_array KMeans<dtype>::
initCentersKMpp(const np_array &np_data, unsigned int random_seed, int n_threads, py::object& callback) const {
    if (np_data.shape(0) < k) {
        std::stringstream ss;
        ss << "not enough data to initialize desired number of centers.";
        ss << "Provided frames (" << np_data.shape(0) << ") < n_centers (" << k << ").";
        throw std::invalid_argument(ss.str());
    }

    constexpr auto size_t_max = std::numeric_limits<std::size_t>::max();

    std::size_t centers_found = 0;
    std::size_t dim = parent_t::metric->dim;

    if (np_data.ndim() != 2) {
        throw std::invalid_argument("input data does not have two dimensions.");
    }

    if (np_data.shape(1) != dim) {
        throw std::invalid_argument("input dimension of data does not match the requested metric ones.");
    }

    auto n_frames = static_cast<size_t>(np_data.shape(0));

    /* number of trials before choosing the data point with the best potential */
    size_t n_trials = 2 + (size_t) log(k);

    /* allocate space for the index giving away which point has already been used as a cluster center */
    std::vector<char> taken_points(n_frames, 0);
    /* candidates allocations */
    std::vector<std::size_t> next_center_candidates(n_trials, size_t_max);
    std::vector<double> next_center_candidates_rand(n_trials, 0);
    std::vector<double> next_center_candidates_potential(n_trials, 0);
    /* allocate space for the array holding the squared distances to the assigned cluster centers */
    std::vector<double> squared_distances(n_frames, 0);

    /* create the output objects */
    std::vector<std::size_t> shape = {k, dim};
    np_array ret_init_centers(shape);
    auto init_centers = ret_init_centers.mutable_unchecked();
    std::memset(init_centers.mutable_data(), 0,
                static_cast<std::size_t>(init_centers.size() * init_centers.itemsize()));

    const auto data = np_data.template unchecked<2>();
    /* initialize random device and pick first center randomly */
    std::default_random_engine generator(random_seed);
    std::uniform_int_distribution<size_t> uniform_dist(0, n_frames - 1);
    auto first_center_index = uniform_dist(generator);
    /* and mark it as assigned */
    taken_points[first_center_index] = 1;
    /* write its coordinates into the init_centers array */
    for (std::size_t j = 0; j < dim; j++) {
        init_centers(centers_found, j) = data(first_center_index, j);
    }
    /* increase number of found centers */
    centers_found++;
    /* perform callback */
    if (!callback.is_none()) {
        py::gil_scoped_acquire acquire;
        callback();
    }
#ifdef USE_OPENMP
    omp_set_num_threads(n_threads);
#endif

    /* iterate over all data points j, measuring the squared distance between j and the initial center i: */
    /* squared_distances[i] = distance(x_j, x_i)*distance(x_j, x_i) */
    dtype dist_sum = 0;
    #pragma omp parallel for reduction(+:dist_sum)
        for (std::size_t i = 0; i < n_frames; i++) {
            if (i != first_center_index) {
                auto value = parent_t::metric->compute(&data(i, 0), &data(first_center_index, 0));
                value = value * value;
                squared_distances[i] = value;
                /* build up dist_sum which keeps the sum of all squared distances */
                dist_sum += value;
            }
        }

    /* keep picking centers while we do not have enough of them... */
    while (centers_found < k) {
        py::gil_scoped_release release;

        /* initialize the trials random values by the D^2-weighted distribution */
        for (std::size_t j = 0; j < n_trials; j++) {
            next_center_candidates[j] = size_t_max;
            auto point_index = uniform_dist(generator);
            next_center_candidates_rand[j] = dist_sum * (static_cast<double>(point_index) /
                                                         static_cast<double>(uniform_dist.max()));
            next_center_candidates_potential[j] = 0.0;
        }
        /* pick candidate data points corresponding to their random value */
        dtype sum = 0.0;
        for (std::size_t i = 0; i < n_frames; i++) {
            if (taken_points[i] == 0) {
                sum += squared_distances[i];
                bool some_not_done{false};
                for (std::size_t j = 0; j < n_trials; j++) {
                    if (next_center_candidates[j] == size_t_max) {
                        if (sum >= next_center_candidates_rand[j]) {
                            assert(i < std::numeric_limits<int>::max());
                            next_center_candidates[j] = i;
                        } else {
                            some_not_done = true;
                        }
                    }
                }
                if (!some_not_done) break;
            }
        }

        /* now find the maximum squared distance for each trial... */
        std::atomic_bool terminate(false);
        #pragma omp parallel for
        for (std::size_t i = 0; i < n_frames; i++) {
            if (terminate.load()) {
                continue;
            }
            if (taken_points.at(i) == 0) {
                for (std::size_t j = 0; j < n_trials; ++j) {
                    if (next_center_candidates.at(j) == size_t_max) {
                        terminate.store(true);
                        break;
                    }
                    #pragma omp critical
                    {
                        if (next_center_candidates.at(j) != i) {
                            auto value = parent_t::metric->compute(&data(i, 0),
                                                                   &data(next_center_candidates.at(j), 0));
                            auto d = value * value;
                            if (d < squared_distances.at(i)) {
                                next_center_candidates_potential[j] += d;
                            } else {
                                next_center_candidates_potential[j] += squared_distances[i];
                            }
                        }
                    }
                }
            }
        }

        /* ... and select the best candidate by the minimum value of the maximum squared distances */
        long best_candidate = -1;
        auto best_potential = std::numeric_limits<double>::max();
        for (std::size_t j = 0; j < n_trials; j++) {
            if (next_center_candidates[j] != size_t_max && next_center_candidates_potential[j] < best_potential) {
                best_potential = next_center_candidates_potential[j];
                best_candidate = next_center_candidates[j];
            }
        }

        /* if for some reason we did not find a best candidate, just take the next available point */
        if (best_candidate == -1) {
            for (std::size_t i = 0; i < n_frames; i++) {
                if (taken_points[i] == 0) {
                    best_candidate = i;
                    break;
                }
            }
        }

        /* check if best_candidate was set, otherwise break to avoid an infinite loop should things go wrong */
        if (best_candidate >= 0) {
            /* write the best_candidate's components into the init_centers array */
            for (std::size_t j = 0; j < dim; j++) {
                init_centers(centers_found, j) = data(best_candidate, j);
            }
            /* increase centers_found */
            centers_found++;
            /* perform the callback */
            if (!callback.is_none()) {
                py::gil_scoped_acquire acquire;
                callback();
            }
            /* mark the data point as assigned center */
            taken_points[best_candidate] = 1;
            /* update the sum of squared distances by removing the assigned center */
            dist_sum -= squared_distances[best_candidate];

            /* if we still have centers to assign, the squared distances array has to be updated */
            if (centers_found < k) {
                /* Check for each data point if its squared distance to the freshly added center is smaller than */
                /* the squared distance to the previously picked centers. If so, update the squared_distances */
                /* array by the new value and also update the dist_sum value by removing the old value and adding */
                /* the new one. */
#pragma omp parallel for
                for (std::size_t i = 0; i < n_frames; ++i) {
                    if (taken_points[i] == 0) {
                        auto value = parent_t::metric->compute(&data(i, 0), &data(best_candidate, 0));
                        auto d = value * value;
#pragma omp critical
                        {
                            if (d < squared_distances[i]) {
                                dist_sum += d - squared_distances[i];
                                squared_distances[i] = d;
                            }
                        }
                    }
                }
            }
        } else {
            break;
        }
    }
    if (centers_found != k) { throw std::runtime_error("kmeans++ failed to initialize all desired centers"); }
    return ret_init_centers;
}

#endif //PYEMMA_KMEANS_BITS_H_H
