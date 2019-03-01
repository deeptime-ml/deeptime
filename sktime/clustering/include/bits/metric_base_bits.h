//
// Created by marscher on 7/21/17.
//

#ifndef PYEMMA_METRIC_BASE_BITS_H
#define PYEMMA_METRIC_BASE_BITS_H

#include "../metric_base.h"

#include <center.h>
#include <theobald_rmsd.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

/**
 * assign a given chunk to given centers using encapsuled metric.
 * @tparam dtype
 * @param chunk
 * @param centers
 * @param n_threads
 * @return
 */
template <typename dtype>
inline py::array_t<int> metric_base<dtype>::assign_chunk_to_centers(const np_array& chunk,
                                                                    const np_array& centers,
                                                                    unsigned int n_threads) {
    if (chunk.ndim() != 2) {
        throw std::invalid_argument("provided chunk does not have two dimensions.");
    }

    if (centers.ndim() != 2) {
        throw std::invalid_argument("provided centers does not have two dimensions.");
    }
    size_t N_centers = static_cast<size_t>(centers.shape(0));
    size_t N_frames = static_cast<size_t>(chunk.shape(0));
    size_t input_dim = static_cast<size_t>(chunk.shape(1));

    if ((input_dim != dim) || (input_dim != centers.shape(1))) {
        throw std::invalid_argument("input dimension mismatch");
    }
    std::vector<dtype> dists(N_centers);
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
                dists[j] = compute(&chunk_buff(i, 0), &centers_buff(j, 0));
            }
            #pragma omp flush(dists)

            /* Only one thread can make actual assignment */
            #pragma omp single
            {
                dtype mindist = std::numeric_limits<dtype>::max();
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

/**
 * euclidean distance method
 * @tparam dtype
 * @param a
 * @param b
 * @return
 */
template <typename dtype>
inline dtype euclidean_metric<dtype>::compute(const dtype *const a, const dtype *const b) {
    double sum = 0.0;
    //#pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < metric_base<dtype>::dim; ++i) {
        assert(std::isfinite(a[i]));
        assert(std::isfinite(b[i]));

        auto d = a[i] - b[i];
        sum += d * d;
    }
    assert(std::isfinite(sum));
    return static_cast<dtype>(std::sqrt(sum));
}

/**
 * minRMSD distance function
 * a: centers
 * b: frames
 * n: dimension of one frame
 * buffer_a: pre-allocated buffer to store a copy of centers
 * buffer_b: pre-allocated buffer to store a copy of frames
 * trace_a_precalc: pre-calculated trace to centers (pointer to one value)
 */
template <typename dtype>
inline dtype min_rmsd_metric<dtype>::compute(const dtype *a, const dtype *b) {
    float trace_a, trace_b;
    auto dim3 = static_cast<const int>(parent_t::dim / 3);
    std::vector<float> buffer_b (b, b + parent_t::dim);

    if (!has_trace_a_been_precalculated) {
        std::vector<float> buffer_a (a, a + parent_t::dim);
        inplace_center_and_trace_atom_major(buffer_a.data(), &trace_a, 1, dim3);
        inplace_center_and_trace_atom_major(buffer_b.data(), &trace_b, 1, dim3);

    } else {
        inplace_center_and_trace_atom_major(buffer_b.data(), &trace_b, 1, dim3);
        trace_a = *trace_centers.data();
    }

    float msd = msd_atom_major(dim3, dim3, a, buffer_b.data(), trace_a, trace_b, 0, nullptr);
    return std::sqrt(msd);
}

template<typename dtype>
inline void min_rmsd_metric<dtype>::precenter_centers(float *centers, std::size_t N_centers) {
    trace_centers.resize(N_centers);
    float *trace_centers_p = trace_centers.data();

    /* Parallelize centering of cluster generators */
    /* Note that this is already OpenMP-enabled */
    for (std::size_t j = 0; j < N_centers; ++j) {
        inplace_center_and_trace_atom_major(&centers[j * parent_t::dim],
                                            &trace_centers_p[j], 1, parent_t::dim / 3);
    }
}


#endif //PYEMMA_METRIC_BASE_BITS_H
