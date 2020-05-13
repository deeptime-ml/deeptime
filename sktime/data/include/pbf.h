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
void forEachParticle(ParticleCollection<dim, dtype> &collection, F op) {

    auto n = collection.nParticles();
    auto *velo = collection.velocities().data();
    auto *pos = collection.positions();

    #pragma omp parallel for default(none), firstprivate(velo, pos, n, op)
    for (std::size_t i = 0; i < n; ++i) {
        op(i, pos + i * dim, velo + i * dim);
    }
}


}

namespace util {
template<int dim, typename dtype>
dtype distance(const dtype *p1, const dtype *p2) {
    dtype d{0};
    for (auto i = 0U; i < dim; ++i) {
        d += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return std::sqrt(d);
}

template<int dim, typename dtype>
dtype dot(const dtype *p1, const dtype *p2) {
    dtype result{0};
    for (auto i = 0u; i < dim; ++i) {
        result += p1[i] * p2[i];
    }
    return result;
}

template<int dim, typename dtype>
dtype length(const dtype *p) {
    dtype l{0};
    for (auto i = 0U; i < dim; ++i) {
        l += p[i] * p[i];
    }
    return std::sqrt(l);
}

template<typename dtype>
dtype Wpoly6(dtype r, dtype h = 2.0) {
    if (r > h)
        return 0;
    float tmp = h * h - r * r;
    return 1.56668147106 * tmp * tmp * tmp / (h * h * h * h * h * h * h * h * h);
}

template<typename dtype>
dtype Wspiky(dtype r, dtype h = 2.0) {
    if (r > h)
        return 0;
    float tmp = h - r;
    return 4.774648292756860 * tmp * tmp * tmp / (h * h * h * h * h * h);
}

template<int dim, typename dtype>
std::array<dtype, dim> gradWspiky(const dtype *pos, const dtype *posNeighbor, dtype h = 2.0) {
    std::array<dtype, dim> result;
    for (auto i = 0U; i < dim; ++i) {
        result[i] = pos[i] - posNeighbor[i];
    }
    dtype l = util::length<dim>(result.data());
    if (l > h || l == 0)
        return std::array<dtype, dim>();
    float tmp = h - l;
    for (auto i = 0u; i < dim; ++i) {
        result[i] = (-3 * 4.774648292756860 * tmp * tmp) * result[i] / (l * h * h * h * h * h * h);
    }
    return result;
}

}

template<int DIM, typename dtype>
class NeighborList {
    static_assert(DIM > 0 && DIM == 2, "Only 2-dimensional NL currently supported.");
public:
    NeighborList(std::array<std::uint32_t, DIM> gridSize, const dtype *const domain, std::size_t nParticles)
            : _gridSize(gridSize), domain(domain), _index(gridSize) {

        for (int i = 0; i < DIM; ++i) {
            if (gridSize[i] == 0) throw std::invalid_argument("grid sizes must be positive.");
            _gridDims[i] = domain[i] / gridSize[i];
        }
        head.resize(_index.size());
        list.resize(nParticles + 1);
    }

    void update(ParticleCollection<DIM, dtype> &collection, int nJobs) {
        auto updateOp = [this](std::size_t particleId, dtype *pos, dtype *vel) {
            auto boxId = positionToBoxIx(pos);

            // CAS
            auto &atomic = *head.at(boxId);
            auto currentHead = atomic.load();
            while (!atomic.compare_exchange_weak(currentHead, particleId)) {}
            list[particleId] = currentHead;
        };

        if (nJobs == 1) {
            for (std::size_t i = 0; i < collection.nParticles(); ++i) {
                updateOp(i, collection.positions() + DIM * i, nullptr);
            }
        } else {
            if (nJobs == 0) {
                nJobs = std::thread::hardware_concurrency();
            }
            std::size_t grainSize = collection.nParticles() / nJobs;
            auto *pptr = collection.positions();
            std::vector<sktime::thread::scoped_thread> jobs;
            for (std::size_t i = 0; i < nJobs - 1; ++i) {
                auto *pptrNext = pptr + grainSize;
                if (pptr != pptrNext) {
                    std::size_t idStart = i * grainSize;
                    jobs.emplace_back([&updateOp, idStart, pptr, pptrNext]() {
                        auto id = idStart;
                        for (auto *p = pptr; p != pptrNext; p += DIM, ++id) {
                            updateOp(id, p, nullptr);
                        }
                    });
                }
                pptr = pptrNext;
            }
            if (pptr != collection.positions() + DIM * collection.nParticles()) {
                auto pptrNext = collection.positions() + DIM * collection.nParticles();
                std::size_t idStart = static_cast<std::size_t>(std::distance(pptr, pptrNext));
                jobs.emplace_back([&updateOp, idStart, pptr, pptrNext]() {
                    auto id = idStart;
                    for (auto *p = pptr; p != pptrNext; p += DIM, ++id) {
                        updateOp(id, p, nullptr);
                    }
                });
            }
        }
    }

    typename Index<DIM>::GridDims gridPos(const dtype *pos) const {
        std::array<std::uint32_t, DIM> projections;
        for (auto i = 0u; i < DIM; ++i) {
            projections[i] = std::floor((pos[i] + .5 * domain[i]) / _gridDims[i]);
        }
        return projections;
    }

    std::uint32_t positionToBoxIx(const dtype *pos) const {
        auto boxId = _index.index(gridPos(pos));
        return boxId;
    }

    template<typename F>
    void forEachNeighbor(std::size_t id, ParticleCollection<DIM, dtype> &collection, F fun) const {
        auto *pos = collection.position(id);
        auto boxId = positionToBoxIx(pos);
        auto neighborId = (*head.at(boxId)).load();
        while (neighborId != 0) {
            if (neighborId != id) {
                fun(neighborId, collection.position(id), collection.velocity(id));
            }
            neighborId = list.at(neighborId);
        }
    }

private:
    std::array<dtype, DIM> _gridDims;
    std::array<std::uint32_t, DIM> _gridSize;
    const dtype *domain;
    std::vector<thread::copyable_atomic<std::size_t>> head;
    std::vector<std::size_t> list;
    Index<DIM> _index;
};

template<int DIM, typename dtype>
class ParticleCollection {
public:
    static constexpr int dim = DIM;

    ParticleCollection(dtype *particles, std::size_t nParticles)
            : _particles(particles), _nParticles(nParticles), _velocities(nParticles * dim, 0) {}

    std::size_t nParticles() const { return _nParticles; }

    std::vector<dtype> &velocities() { return _velocities; }

    const std::vector<dtype> &velocities() const { return _velocities; }

    dtype *const positions() {
        return _particles;
    }

    const dtype *const positions() const {
        return _particles;
    }

    dtype *const position(std::size_t id) {
        return _particles + id * DIM;
    }

    const dtype *const position(std::size_t id) const {
        return _particles + id * DIM;
    }

    dtype *const velocity(std::size_t id) {
        return _velocities.data() + id * DIM;
    }

    const dtype *const velocity(std::size_t id) const {
        return _velocities.data() + id * DIM;
    }

private:
    dtype *const _particles;
    std::vector<dtype> _velocities;
    std::size_t _nParticles;
};

template<int DIM, typename dtype>
class PBF {
public:
    PBF(dtype *particles, std::size_t nParticles, std::array<dtype, DIM> domain,
        std::array<std::uint32_t, DIM> gridSize, int nJobs)
            : _particles(particles, nParticles), _domain(domain), nJobs(nJobs),
              _neighborList(gridSize, _domain.data(), nParticles),
              lambdas(nParticles, 0) {
        detail::setNJobs(nJobs);
    }

    void predictPositions() {
        auto update = [this](std::size_t, dtype *pos, dtype *velocity) {
            velocity[1] += -1 * _gravity * _dt;
            for (auto i = 0u; i < DIM; ++i) {
                pos[i] += _dt * velocity[i];
            }
        };
        detail::forEachParticle(_particles, update);
    }

    void updateNeighborlist() {
        _neighborList.update(_particles, nJobs);
    }

    void calculateLambdas() {
        auto solverOp = [this](std::size_t id, dtype *pos, dtype *vel) {
            dtype sum_k_grad_Ci = 0;
            dtype rho = 0;
            dtype rho0 = this->_rho0;

            std::array<dtype, DIM> grad_pi_Ci;
            _neighborList.forEachNeighbor(id, _particles,
                                          [pos, vel, rho0, &rho, &grad_pi_Ci, &sum_k_grad_Ci](std::size_t neighborId,
                                                                                              dtype *neighborPos,
                                                                                              dtype *neighborVel) {
                                              // compute rho_i (equation 2)
                                              float len = util::distance<DIM>(pos, neighborPos);
                                              float tmp = util::Wpoly6(len);
                                              rho += tmp;

                                              // sum gradients of Ci (equation 8 and parts of equation 9)
                                              // use j as k so that we can stay in the same loop
                                              auto grad_pk_Ci = util::gradWspiky<DIM>(pos, neighborPos);

                                              for (auto i = 0u; i < DIM; ++i) {
                                                  grad_pk_Ci[i] /= rho0;
                                              }
                                              sum_k_grad_Ci += util::dot<DIM>(grad_pk_Ci.data(), grad_pk_Ci.data());

                                              for (auto i = 0u; i < DIM; ++i) {
                                                  grad_pi_Ci[i] += grad_pk_Ci[i];
                                              }
                                          });
            sum_k_grad_Ci += util::dot<DIM>(grad_pi_Ci.data(), grad_pi_Ci.data());

            // compute lambda_i (equations 1 and 9)
            dtype C_i = rho / rho0 - 1;
            dtype lambda = -C_i / (sum_k_grad_Ci + _epsilon);
            lambdas.at(id) = lambda;
        };

        detail::forEachParticle(_particles, solverOp);
    }

    void updatePositions() {
        // todo update positions based on lambdas
    }

    void update() {
        // todo update positions and velocities
    }

    void run(std::uint32_t steps) {
        for (auto step = 0U; step < steps; ++step) {
            predictPositions();
            updateNeighborlist();
            for (auto i = 0u; i < _nSolverIterations; ++i) {
                calculateLambdas();
                updatePositions();
            }
            update();
        }
    }

    void setGravity(dtype gravity) { _gravity = gravity; }

    dtype gravity() const { return _gravity; }

    void setDt(dtype dt) { _dt = dt; }

    dtype dt() const { return _dt; }

    void setRho0(dtype rho0) { _rho0 = rho0; }

    dtype rho0() const { return _rho0; }

    void setEpsilon(dtype epsilon) { _epsilon = epsilon; }

    dtype epsilon() const { return _epsilon; }

    void setNSolverIterations(std::uint32_t n) { _nSolverIterations = n; }

    std::uint32_t nSolverIterations() const { return _nSolverIterations; };

private:
    std::array<dtype, DIM> _domain; // initialize this first as at least neighborList depends on it
    ParticleCollection<DIM, dtype> _particles;
    NeighborList<DIM, dtype> _neighborList;
    std::vector<dtype> lambdas;
    int nJobs;

    std::uint32_t _nSolverIterations = 5;
    dtype _gravity = 10.;
    dtype _dt = 0.016;
    dtype _rho0 = 1.;
    dtype _epsilon = 5.;
};

}
}
