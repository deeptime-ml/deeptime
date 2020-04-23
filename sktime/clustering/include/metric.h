//
// Created by marscher on 3/31/17.
//

#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <vector>

#include "common.h"

namespace py = pybind11;

class Metric {
public:
    virtual ~Metric() = default;
    virtual double compute_d(const double* xs, const double* ys, std::size_t dim) const = 0;
    virtual float compute_f(const float* xs, const float* ys, std::size_t dim) const = 0;

    template<typename T>
    T compute(const T* xs, const T* ys, std::size_t dim) const {
        return compute_d(xs, ys, dim);
    }
};

class EuclideanMetric : public Metric {
public:

    double compute_d(const double *xs, const double *ys, std::size_t dim) const override {
        return _compute(xs, ys, dim);
    }

    float compute_f(const float *xs, const float *ys, std::size_t dim) const override {
        return _compute(xs, ys, dim);
    }

private:
    template<typename T>
    T _compute(const T* xs, const T* ys, std::size_t dim) const {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < dim; ++i) {
            auto d = xs[i] - ys[i];
            sum += d * d;
        }
        return static_cast<T>(std::sqrt(sum));
    }
};

template<typename T>
py::array_t<int> assign_chunk_to_centers(const np_array<T>& chunk,
                                         const np_array<T>& centers,
                                         unsigned int n_threads,
                                         const Metric * metric);



#include "bits/metric_base_bits.h"
