//
// Created by marscher on 3/31/17, adapted by clonker.
//

#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <type_traits>

#include <deeptime/common.h>

namespace deeptime::clustering {

struct EuclideanMetric {
    template<typename dtype>
    static dtype compute(const dtype *xs, const dtype *ys, std::size_t dim) {
        return std::sqrt(compute_squared(xs, ys, dim));
    }

    template<typename dtype>
    static dtype compute_squared(const dtype *xs, const dtype *ys, std::size_t dim) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < dim; ++i) {
            auto d = xs[i] - ys[i];
            sum += d * d;
        }
        return static_cast<dtype>(sum);
    }
};

template<typename Metric, typename T>
py::array_t<int> assign_chunk_to_centers(const np_array_nfc<T> &chunk, const np_array_nfc<T> &centers, int n_threads);

template<typename dtype>
class Distances {
public:
    Distances() : _nXs(0), _nYs(0), _dim(0), _data() {}

    Distances(std::size_t nXs, std::size_t nYs, std::size_t dim) : _nXs(nXs), _nYs(nYs), _dim(dim),
                                                                   _data(new dtype[nXs * nYs]) {}

    const dtype *data() const {
        return _data.get();
    }

    dtype *data() { return _data.get(); }

    np_array<dtype> numpy() const {
        np_array<dtype> result({_nXs, _nYs});
        std::copy(_data.get(), _data.get() + _nXs * _nYs, result.mutable_data());
        return result;
    }

    [[nodiscard]] std::size_t nXs() const { return _nXs; }

    [[nodiscard]] std::size_t nYs() const { return _nYs; }

    [[nodiscard]] std::size_t size() const { return _nXs * _nYs; }

    [[nodiscard]] std::size_t dim() const { return _dim; }

    dtype *begin() { return data(); }

    dtype *end() { return data() + nXs() * nYs(); }

    const dtype *begin() const { return data(); }

    const dtype *end() const { return data() + nXs() * nYs(); }

private:
    std::size_t _nXs, _nYs, _dim;
    std::unique_ptr<dtype[]> _data;
};

template<typename dtype>
std::unique_ptr<double[]> precomputeXX(const dtype *xs, std::size_t nXs, std::size_t dim);

template<bool squared, typename Metric, typename dtype>
Distances<dtype> computeDistances(const dtype *xs, std::size_t nXs,
                                  const dtype *ys, std::size_t nYs, std::size_t dim, const double *xxPrecomputed,
                                  const double *yyPrecomputed);

}

#include "bits/metric_base_bits.h"
