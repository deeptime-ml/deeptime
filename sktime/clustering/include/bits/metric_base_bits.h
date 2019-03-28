//
// Created by marscher on 7/21/17.
//

#pragma once

#include "metric.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

template<typename T>
inline py::array_t<int> assign_chunk_to_centers(const np_array<T>& chunk,
                                                const np_array<T>& centers,
                                                unsigned int n_threads,
                                                const Metric* metric) {
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
    assert(omp_get_num_threads() == n_threads);
#endif
#pragma omp parallel
    {
        for(size_t i = 0; i < N_frames; ++i) {
            /* Parallelize distance calculations to cluster centers to avoid cache misses */
#pragma omp for
            for(size_t j = 0; j < N_centers; ++j) {
                dists[j] = metric->compute(&chunk_buff(i, 0), &centers_buff(j, 0), input_dim);
            }
#pragma omp flush(dists)

            /* Only one thread can make actual assignment */
#pragma omp single
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
            /* Have all threads synchronize in progress through cluster assignments */
#pragma omp barrier
        }
    }
    return dtraj;
}
