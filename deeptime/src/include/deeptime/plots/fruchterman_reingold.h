/*
 * Adapted from networkx.spring_layout (available under BSD3 licence)
 *
 * Copyright (C) 2004-2022, NetworkX Developers
 * Aric Hagberg <hagberg@lanl.gov>
 * Dan Schult <dschult@colgate.edu>
 * Pieter Swart <swart@lanl.gov>
 **/

#pragma once

#include <array>

#include <deeptime/common.h>
#include <deeptime/clustering/metric.h>

namespace deeptime::plots {
template<typename dtype>
deeptime::np_array<dtype> fruchtermanReingold(const deeptime::np_array<dtype> &adjacencyMatrix,
                                              const deeptime::np_array<dtype> &initialPositions,
                                              std::size_t iterations, dtype k,
                                              std::vector<std::size_t> updateDims) {
    static constexpr std::size_t DIM = 2;
    static constexpr std::array<dtype, DIM> ZERO{};
    const auto* zeroPtr = ZERO.data();

    if (updateDims.empty()) {
        updateDims.resize(DIM);
        std::iota(begin(updateDims), end(updateDims), 0);
    }

    const auto nNodes = static_cast<std::size_t>(adjacencyMatrix.shape(0));
    const auto* adjacencyPtr = adjacencyMatrix.data();
    std::vector<dtype> displacement(nNodes * DIM, 0);
    std::vector<dtype> positions(initialPositions.data(0), initialPositions.data(0) + initialPositions.size());

    deeptime::Index<2> positionsIx(std::array<std::size_t, 2>{nNodes, DIM});

    if (k < 0) {
        k = std::sqrt(static_cast<dtype>(1.) / static_cast<dtype>(nNodes));
    }
    std::vector<std::vector<dtype>> sparseAdjacency;

    dtype t{0};
    for (std::size_t d = 0; d < DIM; ++d) {
        dtype max = -std::numeric_limits<dtype>::infinity();
        dtype min = std::numeric_limits<dtype>::infinity();
        for (std::size_t i = 0; i < nNodes; ++i) {
            max = std::max(max, positions[positionsIx(i, d)]);
            min = std::min(min, positions[positionsIx(i, d)]);
        }
        t = std::max(t, max - min);
    }
    t *= .1;
    const auto dt = t / static_cast<dtype>(iterations + 1);

    std::vector<dtype> delta(nNodes * nNodes * DIM, 0);
    deeptime::Index<3> deltaIx(std::array<std::size_t, 3>{nNodes, nNodes, DIM});
    std::vector<dtype> distances(nNodes * nNodes, 0);
    deeptime::Index<2> distancesIx(std::array<std::size_t, 2>{nNodes, nNodes});

    std::vector<dtype> length(nNodes, 0);

    for (std::size_t iter = 0; iter < iterations; ++iter) {
        std::fill(begin(displacement), end(displacement), 0);

        {
            // fill up differences (delta) and distances
            #pragma omp parallel for collapse(2) default(none) firstprivate(nNodes, zeroPtr) \
                        shared(delta, deltaIx, positions, positionsIx, distances, distancesIx)
            for (std::size_t i = 0; i < nNodes; ++i) {
                for (std::size_t j = 0; j < nNodes; ++j) {
                    for (std::size_t d = 0; d < DIM; ++d) {
                        delta[deltaIx(i, j, d)] = positions[positionsIx(i, d)] - positions[positionsIx(j, d)];
                    }
                    const auto norm = deeptime::clustering::EuclideanMetric::compute(
                            &delta[deltaIx(i, j, 0)], zeroPtr, DIM
                    );
                    distances[distancesIx(i, j)] = std::max(static_cast<dtype>(0.1), norm);
                }
            }
        }

        {
            // compute displacement and lengths
            #pragma omp parallel for default(none) collapse(2) firstprivate(nNodes, k) \
                shared(distances, adjacencyPtr, distancesIx, displacement, delta, positionsIx, deltaIx)
            for (std::size_t i = 0; i < nNodes; ++i) {
                for (std::size_t j = 0; j < nNodes; ++j) {
                    const auto &dist = distances[distancesIx(i, j)];
                    const auto &adj = adjacencyPtr[distancesIx(i, j)];
                    const auto force = k * k / (dist * dist) - adj * dist / k;
                    for (std::size_t d = 0; d < DIM; ++d) {
                        displacement[positionsIx(i, d)] += delta[deltaIx(i, j, d)] * force;
                    }
                }
            }

            #pragma omp parallel for default(none) firstprivate(nNodes, zeroPtr, DIM) \
                shared(displacement, positionsIx, length)
            for (std::size_t i = 0; i < nNodes; ++i) {
                const auto d = deeptime::clustering::EuclideanMetric::compute(
                        &displacement[positionsIx(i, 0)], zeroPtr, DIM
                );
                length[i] = d < .01 ? static_cast<dtype>(0.1) : d;
            }

            #pragma omp parallel for default(none) firstprivate(nNodes, t) \
                shared(displacement, positions, length, positionsIx, updateDims)
            for (std::size_t i = 0; i < nNodes; ++i) {
                for (const auto &d : updateDims) {
                    displacement[positionsIx(i, d)] *= t / length[i];
                    positions[positionsIx(i, d)] += displacement[positionsIx(i, d)];
                }
            }
        }
        t -= dt;

    }

    deeptime::np_array<dtype> output(std::array<std::size_t, 2>{nNodes, DIM});
    std::copy(begin(positions), end(positions), output.mutable_data(0));

    return output;
}
}
