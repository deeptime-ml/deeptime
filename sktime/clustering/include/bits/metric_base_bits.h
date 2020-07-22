/********************************************************************************
 * This file is part of scikit-time.                                            *
 *                                                                              *
 * Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)         *
 *                                                                              *
 * scikit-time is free software: you can redistribute it and/or modify          *
 * it under the terms of the GNU Lesser General Public License as published by  *
 * the Free Software Foundation, either version 3 of the License, or            *
 * (at your option) any later version.                                          *
 *                                                                              *
 * This program is distributed in the hope that it will be useful,              *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
 * GNU General Public License for more details.                                 *
 *                                                                              *
 * You should have received a copy of the GNU Lesser General Public License     *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.        *
 ********************************************************************************/

//
// Created by marscher on 7/21/17, adapted by clonker.
//

#pragma once

#include "metric.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

template<>
inline float Metric::compute_squared<float>(const float* xs, const float* ys, std::size_t dim) const {
    return compute_squared_f(xs, ys, dim);
}

template<>
inline double Metric::compute_squared<double>(const double* xs, const double* ys, std::size_t dim) const {
    return compute_squared_d(xs, ys, dim);
}

template<typename T>
inline py::array_t<int> assign_chunk_to_centers(const np_array<T>& chunk,
                                                const np_array<T>& centers,
                                                int n_threads,
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
#endif

    #pragma omp parallel default(none) firstprivate(N_frames, N_centers, centers_buff, input_dim, metric, chunk_buff, dtraj_buff, dists)
    {
        #pragma omp for
        for(size_t i = 0; i < N_frames; ++i) {
            for(size_t j = 0; j < N_centers; ++j) {
                dists[j] = metric->compute(&chunk_buff(i, 0), &centers_buff(j, 0), input_dim);
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
