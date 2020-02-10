//
// Created by mho on 2/10/20.
//

#pragma once

#include "common.h"

template<typename dtype>
np_array<int> viterbi(const np_array<dtype> &transitionMatrix, const np_array<dtype> &stateProbabilityTraj,
                      const np_array<dtype> &initialDistribution) {
    auto N = static_cast<std::size_t>(transitionMatrix.shape(0));
    auto T = static_cast<std::size_t>(stateProbabilityTraj.shape(0));
    np_array<std::int32_t> path(std::vector<std::size_t>{T});
    auto pathBuf = path.mutable_data();
    auto ABuf = transitionMatrix.data();
    auto pobsBuf = stateProbabilityTraj.data();
    auto piBuf = initialDistribution.data();
    {
        py::gil_scoped_release gil;

        std::fill(pathBuf, pathBuf + path.size(), 0);

        std::size_t i, j, t, maxi;
        dtype sum;
        auto vData = std::unique_ptr<dtype[]>(new dtype[N]);
        auto v = vData.get();
        auto vnextData = std::unique_ptr<dtype[]>(new dtype[N]);
        auto vnext = vnextData.get();
        auto hData = std::unique_ptr<dtype[]>(new dtype[N]);
        auto h = hData.get();
        auto ptr = std::unique_ptr<std::int32_t[]>(new std::int32_t[T * N]);

        // initialization of v
        sum = 0.0;
        for (i = 0; i < N; i++) {
            v[i] = pobsBuf[i] * piBuf[i];
            sum += v[i];
        }
        // normalize
        for (i = 0; i < N; i++) {
            v[i] /= sum;
        }

        // iteration of v
        for (t = 1; t < T; t++) {
            sum = 0.0;
            for (j = 0; j < N; j++) {
                for (i = 0; i < N; i++) {
                    h[i] = v[i] * ABuf[i * N + j];
                }
                maxi = std::distance(h, std::max_element(h, h + N));
                ptr[t * N + j] = maxi;
                vnext[j] = pobsBuf[t * N + j] * v[maxi] * ABuf[maxi * N + j];
                sum += vnext[j];
            }
            // normalize
            for (i = 0; i < N; i++) {
                vnext[i] /= sum;
            }
            // update v
            std::swap(v, vnext);
        }

        // path reconstruction
        pathBuf[T - 1] = std::distance(v, std::max_element(v, v + N));
        for (t = T - 1; t >= 1; t--) {
            pathBuf[t - 1] = ptr[t * N + pathBuf[t]];
        }


    }
    return path;
}
