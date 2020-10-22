//
// Created by marscher on 3/31/17, adapted by clonker.
//

#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <vector>

#include "common.h"

class Metric {
public:
    virtual ~Metric() = default;
    virtual double compute_squared_d(const double* xs, const double* ys, std::size_t dim) const = 0;
    virtual float compute_squared_f(const float* xs, const float* ys, std::size_t dim) const = 0;

    virtual bool isEuclidean() const {
        return false;
    }

    template<typename T>
    T compute(const T* xs, const T* ys, std::size_t dim) const {
        return std::sqrt(compute_squared(xs, ys, dim));
    }

    template<typename T>
    T compute_squared(const T* xs, const T* ys, std::size_t dim) const;
};

class EuclideanMetric : public Metric {
public:

    double compute_squared_d(const double *xs, const double *ys, std::size_t dim) const override {
        return _compute_squared(xs, ys, dim);
    }

    float compute_squared_f(const float *xs, const float *ys, std::size_t dim) const override {
        return _compute_squared(xs, ys, dim);
    }

    bool isEuclidean() const override {
        return true;
    }

private:
    template<typename T>
    T _compute_squared(const T* xs, const T* ys, std::size_t dim) const {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < dim; ++i) {
            auto d = xs[i] - ys[i];
            sum += d * d;
        }
        return static_cast<T>(sum);
    }
};

inline static const EuclideanMetric* default_metric(){
    static thread_local EuclideanMetric instance = EuclideanMetric{};
    return &instance;
}

template<typename T>
py::array_t<int> assign_chunk_to_centers(const np_array_nfc<T>& chunk,
                                         const np_array_nfc<T>& centers,
                                         int n_threads,
                                         const Metric * metric);

template<typename dtype>
class Distances {
public:
    Distances() : _nXs(0), _nYs(0), _dim(0), _data() {}
    Distances(std::size_t nXs, std::size_t nYs, std::size_t dim) : _nXs(nXs), _nYs(nYs), _dim(dim),
                                                                   _data(new dtype[nXs * nYs]) {}

    const dtype* data() const {
        return _data.get();
    }

    dtype* data() { return _data.get(); }

    np_array<dtype> numpy() const {
        np_array<dtype> result ({_nXs, _nYs});
        std::copy(_data.get(), _data.get() + _nXs * _nYs, result.mutable_data());
        return result;
    }

    std::size_t nXs() const { return _nXs; }
    std::size_t nYs() const { return _nYs; }
    std::size_t size() const { return _nXs * _nYs; }
    std::size_t dim() const { return _dim; }

    dtype* begin() { return data(); }
    dtype* end() { return data() + nXs() * nYs(); }

    const dtype* begin() const { return data(); }
    const dtype* end() const { return data() + nXs() * nYs(); }

private:
    std::size_t _nXs, _nYs, _dim;
    std::unique_ptr<dtype[]> _data;
};

template<typename dtype>
std::unique_ptr<double[]> precomputeXX(const dtype* xs, std::size_t nXs, std::size_t dim) {
    std::unique_ptr<double[]> xx(new double[nXs]);
    auto xp = xx.get();

#pragma omp parallel for
    for (std::size_t i = 0; i < nXs; ++i) {
        xp[i] = std::inner_product(xs + i*dim, xs + i*dim + dim, xs + i*dim, static_cast<double>(0));
    }
    return xx;
}

template<bool squared, typename dtype>
Distances<dtype> computeDistances(const dtype* xs, std::size_t nXs,
                                  const dtype* ys, std::size_t nYs, std::size_t dim, const double* xxPrecomputed, const double* yyPrecomputed,
                                  const Metric * metric) {
    Distances<dtype> result (nXs, nYs, dim);
    if(!metric->isEuclidean()) {
        dtype* outPtr = result.data();
        if (squared) {
            #pragma omp parallel for default(none) firstprivate(nXs, nYs, xs, ys, dim, metric, outPtr)
            for (std::size_t i = 0; i < nXs; ++i) {
                for (std::size_t j = 0; j < nYs; ++j) {
                    outPtr[i * nYs + j] = metric->compute_squared(xs + i * dim, ys + j * dim, dim);
                }
            }
        } else {
            #pragma omp parallel for default(none) firstprivate(nXs, nYs, xs, ys, dim, metric, outPtr)
            for (std::size_t i = 0; i < nXs; ++i) {
                for (std::size_t j = 0; j < nYs; ++j) {
                    outPtr[i * nYs + j] = metric->compute(xs + i * dim, ys + j * dim, dim);
                }
            }
        }
    } else {
        dtype* outPtr = result.data();
        // xxPrecomputed has shape (nXs,)
        // yyPrecomputed has shape (nYs,)
        std::unique_ptr<double[]> xx;
        if(xxPrecomputed == nullptr) {
            xx = precomputeXX(xs, nXs, dim);
            xxPrecomputed = xx.get();
        }
        std::unique_ptr<double[]> yy;
        if(yyPrecomputed == nullptr) {
            yy = precomputeXX(ys, nYs, dim);
            yyPrecomputed = yy.get();
        }
        {
            // compute -2 * XY
            #pragma omp parallel for collapse(2)
            for (std::size_t i = 0; i < nXs; ++i) {
                for (std::size_t j = 0; j < nYs; ++j) {
                    outPtr[i * nYs + j] = std::inner_product(xs + i * dim, xs + i*dim + dim, ys + j*dim, static_cast<double>(0));
                    outPtr[i * nYs + j] *= -2.;
                    outPtr[i * nYs + j] += xxPrecomputed[i] + yyPrecomputed[j];
                }
            }
        }
        if (!squared) {
            #pragma omp parallel for
            for(std::size_t i = 0; i < result.size(); ++i) {
                *(result.data() + i) = std::sqrt(*(result.data() + i));
            }
        }
    }
    return result;
}



#include "bits/metric_base_bits.h"
