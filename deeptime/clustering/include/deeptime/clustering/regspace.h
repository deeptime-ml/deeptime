//
// Created by marscher on 7/17/17.
//

#pragma once

#if defined(USE_OPENMP)
#include <omp.h>
#endif

#include <deeptime/common.h>
#include "metric.h"

namespace deeptime::clustering::regspace {

class MaxCentersReachedException : public std::exception {
public:
    explicit MaxCentersReachedException(const char *m) : message{m} {}

    [[nodiscard]] const char *what() const noexcept override { return message.c_str(); }

private:
    std::string message;
};

/**
 * loops over all points in chunk and checks for each center if the distance is smaller than dmin,
 * if so, the point is appended to py_centers. This is done until max_centers is reached or all points have been
 * added to the list.
 * @param chunk array shape(n, d)
 * @param py_centers python list containing found centers.
 */
template<typename Metric, typename T>
void cluster(const np_array_nfc<T> &chunk, py::list& py_centers, T dmin, std::size_t maxClusters, int n_threads);

}

#include "./bits/regspace_bits.h"
