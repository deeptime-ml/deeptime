//
// Created by marscher on 4/3/17.
//

#pragma once

#include <utility>

#include "common.h"
#include "metric.h"

namespace clustering {
namespace kmeans {

template<typename T>
np_array<T> cluster(const np_array<T> & /*np_chunk*/, const np_array<T> & /*np_centers*/, int /*n_threads*/,
                    const Metric *metric);
template<typename T>
std::tuple<np_array<T>, int, int, T> cluster_loop(const np_array<T>& np_chunk, const np_array<T>& np_centers,
                                                  std::size_t k, const Metric *metric,
                                                  int n_threads, int max_iter, T tolerance,
                                                  py::object& callback);
template<typename T>
T costFunction(const np_array<T>& np_data, const np_array<T>& np_centers, const Metric *metric, int n_threads);

template<typename T>
np_array<T> initCentersKMpp(const np_array<T>& np_data, std::size_t k, const Metric *metric,
                            std::int64_t random_seed, int n_threads, py::object& callback);

}
}
#include "bits/kmeans_bits.h"
