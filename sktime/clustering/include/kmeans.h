//
// Created by marscher on 4/3/17.
//

#pragma once

#include <utility>

#include "common.h"

namespace clustering::kmeans {

template<typename T, typename MetricFunc>
np_array<T> cluster(const np_array<T> & /*np_chunk*/, const np_array<T> & /*np_centers*/, int /*n_threads*/,
                    const MetricFunc &metric);
template<typename T, typename MetricFunc>
std::tuple<np_array<T>, int, int> cluster_loop(const np_array<T>& np_chunk, np_array<T>& np_centers,
                                               std::size_t k, const MetricFunc &metric,
                                               int n_threads, int max_iter, T tolerance,
                                               py::object& callback);
template<typename T, typename MetricFunc>
T costFunction(const np_array<T>& np_data, const np_array<T>& np_centers, const MetricFunc &metric, int n_threads);

template<typename T, template<typename> typename MetricFunc>
np_array<T> initCentersKMpp(const np_array<T>& np_data, std::size_t k, const MetricFunc<T> &metric,
                            unsigned int random_seed, int n_threads, py::object& callback);

}

#include "bits/kmeans_bits.h"
