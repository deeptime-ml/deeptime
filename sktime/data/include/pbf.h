/**
 *
 *
 * @file pbf.h
 * @brief 
 * @author clonker
 * @date 5/13/20
 */
#pragma once

#include <algorithm>
#include <atomic>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "thread_utils.h"

namespace sktime {
namespace pbf {

template<int DIM, typename dtype>
class ParticleCollection;

namespace detail {
inline void setNJobs(int nJobs) {
    #ifdef USE_OPENMP
    omp_set_num_threads(nJobs);
    #endif
}

template<int dim, typename dtype, typename F>
void forEachParticle(ParticleCollection<dim, dtype>& collection, F op) {

    auto n = collection.nParticles();
    auto* velo = collection.velocities().data();
    auto* pos = collection.positions();

    #pragma omp parallel for default(none), firstprivate(velo, pos, n, op)
    for(std::size_t i = 0; i < n; ++i) {
        op(i, pos + i*dim, velo + i*dim);
    }
}

}

template<int DIM, typename dtype>
class NeighborList {
    static_assert(DIM > 0 && DIM == 2, "Only 2-dimensional NL currently supported.");
public:
    NeighborList(std::array<std::uint32_t, DIM> gridSize, const dtype* const domain, std::size_t nParticles)
        : _gridSize(gridSize), domain(domain), _index(gridSize){

        for(int i = 0; i < DIM; ++i) {
            if(gridSize[i] == 0) throw std::invalid_argument("grid sizes must be positive.");
            _gridDims[i] = domain[i] / gridSize[i];
        }
        head.resize(_index.size());
        list.resize(nParticles + 1);
    }

    void update(ParticleCollection<DIM, dtype> &collection, int nJobs) {
        auto updateOp = [this](std::size_t particleId, dtype* pos, dtype* vel) {
            std::array<std::uint32_t, DIM> projections;
            for(auto i = 0u; i < DIM; ++i) {
                projections[i] = std::floor((pos[i] + .5 * domain[i]) / _gridDims[i]);
            }
            auto boxId = _index.index(projections);

            // CAS
            auto &atomic = head.at(boxId);
            auto currentHead = atomic.load();
            while (!atomic.compare_exchange_weak(currentHead, particleId)) {}
            list[particleId] = currentHead;
        };


    }

private:
    std::array<dtype, DIM> _gridDims;
    std::array<std::uint32_t, DIM> _gridSize;
    dtype* const domain;
    std::vector<std::atomic<std::size_t>> head;
    std::vector<std::size_t> list;
    Index<DIM> _index;
};

template<int DIM, typename dtype>
class ParticleCollection {
public:
    static constexpr int dim = DIM;

    ParticleCollection(dtype* particles, std::size_t nParticles)
        : _particles(particles), _nParticles(nParticles), _velocities(nParticles*dim, 0) {}

    std::size_t nParticles() const { return _nParticles; }
    std::vector<dtype>& velocities() { return _velocities; }
    const std::vector<dtype>& velocities() const { return _velocities; }

    dtype* const positions() {
        return _particles;
    }

    const dtype* const positions() const {
        return _particles;
    }

private:
    dtype* const _particles;
    std::vector<dtype> _velocities;
    std::size_t _nParticles;
};

template<int DIM, typename dtype>
class PBF {
public:
    PBF(dtype* particles, std::size_t nParticles, std::array<dtype, DIM> domain, dtype gravity, dtype dt, int nJobs)
        : _particles(particles, nParticles), _gravity(gravity), _dt(dt), _domain(domain) {
        detail::setNJobs(nJobs);
    }

    void predictPositions() {
        auto update = [this](std::size_t, dtype* pos, dtype* velocity) {
            velocity[1] += -1 * _gravity * _dt;
            for(int i = 0; i < DIM; ++i) {
                pos[i] += _dt * velocity[i];
            }
        };
        detail::forEachParticle(_particles, update);
    }



private:
    ParticleCollection<DIM, dtype> _particles;
    std::array<dtype, DIM> _domain;
    dtype _gravity;
    dtype _dt;
};

template<int dim, typename dtype>
void predictPosition(dtype* pos, dtype* velocity) {

}

}
}
