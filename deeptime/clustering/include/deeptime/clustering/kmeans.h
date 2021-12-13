//
// Created by marscher on 4/3/17, adapted by clonker.
//

#pragma once

#include <utility>
#include <random>
#include <atomic>

#include <pybind11/pytypes.h>
#include <mutex>

#include <deeptime/common.h>
#include "metric.h"
#include <deeptime/util/thread_utils.h>
#include <deeptime/util/distribution_utils.h>

namespace deeptime::clustering::kmeans {

template<typename Metric, typename T>
std::tuple<np_array<T>, np_array<int>> cluster(const np_array_nfc<T>& np_chunk, const np_array_nfc<T>& np_centers,
                                               int n_threads);

template<typename Metric, typename T>
std::tuple<np_array_nfc<T>, int, int, np_array<T>> cluster_loop(
        const np_array_nfc<T> &np_chunk, const np_array_nfc<T> &np_centers,
        int n_threads, int max_iter, T tolerance, py::object &callback);


template<typename Metric, typename T>
T costFunction(const np_array_nfc<T> &np_data, const np_array_nfc<T> &np_centers,
               const np_array<int> &assignments, int n_threads);

template<typename Metric, typename T>
T costAssignFunction(const np_array_nfc<T> &np_data, const np_array_nfc<T> &np_centers, int n_threads);

namespace util {
template<typename dtype, typename itype>
void assignCenter(itype frameIndex, std::size_t dim, const dtype * data, dtype * centers);
}

template<typename Metric, typename dtype>
np_array<dtype> initKmeansPlusPlus(const np_array_nfc<dtype> &data, std::size_t k,
                                   std::int64_t seed, int n_threads, py::object &callback);

}
#include "bits/kmeans_bits.h"
