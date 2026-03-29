//
// Created by marscher on 7/21/17, adapted by clonker.
//

#pragma once

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace deeptime::clustering {

template<typename Metric, typename T>
inline py::array_t<int> assign_chunk_to_centers(const np_array_nfc <T> &chunk,
                                                const np_array_nfc <T> &centers,
                                                int n_threads) {
    if (chunk.ndim() != 2) {
        throw std::invalid_argument("provided chunk does not have two dimensions.");
    }

    if (centers.ndim() != 2) {
        throw std::invalid_argument("provided centers does not have two dimensions.");
    }

    if (chunk.shape(1) != centers.shape(1)) {
        throw std::invalid_argument("dimension mismatch centers and provided data to assign.");
    }

    auto N_centers = static_cast<size_t>(centers.shape(0));
    auto N_frames = static_cast<size_t>(chunk.shape(0));
    auto input_dim = static_cast<size_t>(chunk.shape(1));

    std::vector<T> dists(N_centers);
    std::vector<size_t> shape = {N_frames};
    py::array_t<int> dtraj(shape);

    auto dtraj_buff = dtraj.template mutable_unchecked<1>();
    auto chunk_buff = chunk.template unchecked<2>();
    auto centers_buff = centers.template unchecked<2>();

#ifdef USE_OPENMP
    /* Create a parallel thread block. */
    omp_set_num_threads(n_threads);
#endif

    #pragma omp parallel default(none) firstprivate(N_frames, N_centers, centers_buff, input_dim, chunk_buff, dtraj_buff, dists)
    {
        #pragma omp for
        for (size_t i = 0; i < N_frames; ++i) {
            for (size_t j = 0; j < N_centers; ++j) {
                dists[j] = Metric::template compute<T>(&chunk_buff(i, 0), &centers_buff(j, 0), input_dim);
            }

            {
                T mindist = std::numeric_limits<T>::max();
                int argmin = -1;
                for (size_t j = 0; j < N_centers; ++j) {
                    if (dists[j] < mindist) {
                        mindist = dists[j];
                        argmin = static_cast<int>(j);
                    }
                }
                dtraj_buff(i) = argmin;
            }
        }
    }
    return dtraj;
}

template<typename dtype>
inline std::unique_ptr<double[]> precomputeXX(const dtype *xs, std::size_t nXs, std::size_t dim) {
    std::unique_ptr<double[]> xx(new double[nXs]);
    auto xp = xx.get();

    #pragma omp parallel for default(none) firstprivate(xs, nXs, dim, xp)
    for (std::size_t i = 0; i < nXs; ++i) {
        xp[i] = std::inner_product(xs + i * dim, xs + i * dim + dim, xs + i * dim, static_cast<double>(0));
    }
    return xx;
}

template<bool squared, typename Metric, typename dtype>
inline Distances<dtype> computeDistances(const dtype *xs, std::size_t nXs,
                                  const dtype *ys, std::size_t nYs, std::size_t dim, const double *xxPrecomputed,
                                  const double *yyPrecomputed) {
    Distances<dtype> result(nXs, nYs, dim);
    if constexpr (!std::is_same<Metric, EuclideanMetric>::value) {
        dtype *outPtr = result.data();
        #pragma omp parallel for default(none) firstprivate(nXs, nYs, xs, ys, dim, outPtr)
        for (std::size_t i = 0; i < nXs; ++i) {
            for (std::size_t j = 0; j < nYs; ++j) {
                if constexpr(squared) {
                    outPtr[i * nYs + j] = Metric::template compute_squared<dtype>(xs + i * dim, ys + j * dim, dim);
                } else {
                    outPtr[i * nYs + j] = Metric::template compute<dtype>(xs + i * dim, ys + j * dim, dim);
                }
            }
        }
    } else {
        dtype *outPtr = result.data();
        // xxPrecomputed has shape (nXs,)
        // yyPrecomputed has shape (nYs,)
        std::unique_ptr<double[]> xx;
        if (xxPrecomputed == nullptr) {
            xx = precomputeXX(xs, nXs, dim);
            xxPrecomputed = xx.get();
        }
        std::unique_ptr<double[]> yy;
        if (yyPrecomputed == nullptr) {
            yy = precomputeXX(ys, nYs, dim);
            yyPrecomputed = yy.get();
        }
        {
            // compute -2 * XY
            #pragma omp parallel for default(none) firstprivate(nXs, nYs, xs, dim, ys, xxPrecomputed, yyPrecomputed, outPtr) collapse(2)
            for (std::size_t i = 0; i < nXs; ++i) {
                for (std::size_t j = 0; j < nYs; ++j) {
                    outPtr[i * nYs + j] = std::inner_product(xs + i * dim, xs + i * dim + dim, ys + j * dim,
                                                             static_cast<double>(0));
                    outPtr[i * nYs + j] *= -2.;
                    outPtr[i * nYs + j] += xxPrecomputed[i] + yyPrecomputed[j];
                }
            }
        }
        if (!squared) {
            auto size = result.size();
            auto* ptr = result.data();
            #pragma omp parallel for default(none) firstprivate(size, ptr)
            for (std::size_t i = 0; i < size; ++i) {
                ptr[i] = std::sqrt(ptr[i]);
            }
        }
    }
    return result;
}

}
