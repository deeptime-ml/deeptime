// author: clonker

#pragma once

#include <algorithm>
#include <atomic>

#ifdef USE_OPENMP

#include <omp.h>

#endif

#include "thread_utils.h"

namespace deeptime::pbf {

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
template<class T>
constexpr const T &clamp(const T &v, const T &lo, const T &hi) {
    assert(!(hi < lo));
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

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
dtype Wpoly6(dtype r, dtype h) {
    if (r > h)
        return 0;
    float tmp = h * h - r * r;
    return 1.56668147106 * tmp * tmp * tmp / (h * h * h * h * h * h * h * h * h);
}

template<typename dtype>
dtype Wspiky(dtype r, dtype h) {
    if (r > h)
        return 0;
    float tmp = h - r;
    return 4.774648292756860 * tmp * tmp * tmp / (h * h * h * h * h * h);
}

template<int dim, typename dtype>
std::array<dtype, dim> gradWspiky(const dtype *pos, const dtype *posNeighbor, dtype h) {
    std::array<dtype, dim> result;
    for (auto i = 0U; i < dim; ++i) {
        result[i] = pos[i] - posNeighbor[i];
    }
    dtype l = util::length<dim>(result.data());
    if (l > h || l == 0) {
        std::fill(result.begin(), result.end(), 0);
        return result;
    }
    float tmp = h - l;
    for (auto i = 0u; i < dim; ++i) {
        result[i] = (-3 * 4.774648292756860 * tmp * tmp) * result[i] / (l * h * h * h * h * h * h);
    }
    return result;
}

}

template<int DIM, typename dtype>
class NeighborList {
    static_assert(DIM == 2, "Only 2-dimensional NL currently supported.");
public:
    NeighborList() : _gridSize() {}

    NeighborList(std::array<dtype, DIM> gridSize, dtype interactionRadius, std::size_t nParticles)
            : _gridSize(gridSize) {
        for (int i = 0; i < DIM; ++i) {
            _cellSize[i] = interactionRadius;
            if (gridSize[i] <= 0) throw std::invalid_argument("grid sizes must be positive.");
            nCells[i] = gridSize[i] / _cellSize[i];
        }
        _index = Index<DIM>(nCells);
        head.resize(_index.size());
        list.resize(nParticles + 1);
    }

    void update(ParticleCollection<DIM, dtype> &collection, int nJobs) {
        std::fill(std::begin(list), std::end(list), 0);
        std::fill(std::begin(head), std::end(head), thread::copyable_atomic<std::size_t>());
        auto updateOp = [this](std::size_t particleId, dtype *pos, dtype * /*velocity*/ ) {
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
                nJobs = static_cast<decltype(nJobs)>(std::thread::hardware_concurrency());
            }
            if (nJobs <= 1) {
                throw std::logic_error("At this point nJobs should be >= 2");
            }
            std::size_t grainSize = collection.nParticles() / nJobs;
            auto *pptr = collection.positions();
            std::vector<deeptime::thread::scoped_thread> jobs;
            for (int i = 0; i < nJobs - 1; ++i) {
                auto *pptrNext = std::min(pptr + grainSize * DIM,
                                          collection.positions() + collection.nParticles() * DIM);
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
                std::size_t idStart = (nJobs - 1) * grainSize;
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
            projections[i] = static_cast<std::uint32_t>(
                    std::max(static_cast<dtype>(0.), static_cast<dtype>(std::floor((pos[i] + .5 * _gridSize[i]) / _cellSize[i]))));
            projections[i] = util::clamp(projections[i], 0U, nCells[i] - 1);
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
        auto gridPos = this->gridPos(pos);
        for (int i = 0u; i < DIM; ++i) {
            std::uint32_t begin = gridPos[i] > 0 ? (gridPos[i] - 1) : gridPos[i];
            std::uint32_t end = gridPos[i] < nCells[i] - 1 ? (gridPos[i] + 2) : gridPos[i] + 1;

            auto cellPos = gridPos;
            for (auto k = begin; k < end; ++k) {
                cellPos[i] = k;

                if (cellPos != gridPos) {
                    auto cellId = _index.index(cellPos);
                    auto neighborId = (*head.at(cellId)).load();
                    while (neighborId != 0) {
                        fun(neighborId, collection.position(neighborId), collection.velocity(neighborId));
                        neighborId = list.at(neighborId);
                    }
                }
            }
        }
        {
            auto boxId = positionToBoxIx(pos);
            auto neighborId = (*head.at(boxId)).load();
            while (neighborId != 0) {
                if (neighborId != id) {
                    fun(neighborId, collection.position(neighborId), collection.velocity(neighborId));
                }
                neighborId = list.at(neighborId);
            }
        }
    }

    const std::array<dtype, DIM> &gridSize() const {
        return _gridSize;
    }

private:
    std::array<dtype, DIM> _cellSize{};
    std::array<dtype, DIM> _gridSize{};
    std::vector<thread::copyable_atomic<std::size_t>> head{};
    std::vector<std::size_t> list{};
    Index<DIM> _index{};
    std::array<std::uint32_t, DIM> nCells{};
};

template<int DIM, typename dtype>
class ParticleCollection {
public:
    static constexpr int dim = DIM;

    ParticleCollection() = default;

    ParticleCollection(std::vector<dtype> particles, std::size_t nParticles)
            : _nParticles(nParticles), _particles(std::move(particles)), _velocities(nParticles * dim, 0) {
    }

    std::size_t nParticles() const { return _nParticles; }

    std::vector<dtype> &velocities() { return _velocities; }

    const std::vector<dtype> &velocities() const { return _velocities; }

    dtype * positions() {
        return _particles.data();
    }

    const dtype * positions() const {
        return _particles.data();
    }

    dtype * position(std::size_t id) {
        return _particles.data() + id * DIM;
    }

    const dtype * position(std::size_t id) const {
        return _particles.data() + id * DIM;
    }

    dtype * velocity(std::size_t id) {
        return _velocities.data() + id * DIM;
    }

    const dtype * velocity(std::size_t id) const {
        return _velocities.data() + id * DIM;
    }


private:
    std::size_t _nParticles{};
    std::vector<dtype> _particles{};
    std::vector<dtype> _velocities{};
};

template<int DIM, typename dtype>
class PBF {
public:
    PBF() : nJobs(0), _interactionRadius(0), _particles(nullptr, 0), _neighborList(), _positions(), lambdas() {}

    PBF(std::vector<dtype> particles, std::size_t nParticles, std::array<dtype, DIM> gridSize,
        dtype interactionRadius, int nJobs)
            : nJobs(nJobs), _interactionRadius(interactionRadius), _positions(nParticles * DIM, 0),
              lambdas(nParticles, 0),
              _particles(std::move(particles), nParticles),
              _neighborList(gridSize, interactionRadius, nParticles) {
        detail::setNJobs(nJobs);
        std::copy(_particles.positions(), _particles.positions() + nParticles * DIM, _positions.data());
    }

    PBF(PBF &&) = default;

    PBF &operator=(PBF &&) = default;

    PBF(const PBF &) = delete;

    virtual ~PBF() = default;

    PBF &operator=(const PBF &) = delete;

    void predictPositions(dtype drift) {
        auto update = [this, drift](std::size_t, dtype *pos, dtype *velocity) {
            velocity[1] += -1 * _gravity * _dt;
            velocity[0] -= drift * _gravity * _dt;
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
        auto solverOp = [this](std::size_t id, dtype *pos, dtype */*vel*/) {
            dtype sum_k_grad_Ci = 0;
            dtype rho = 0;
            dtype rho0 = this->_rho0;
            dtype h = this->_interactionRadius;

            std::array<dtype, DIM> grad_pi_Ci{};
            std::fill(grad_pi_Ci.begin(), grad_pi_Ci.end(), 0);
            auto neighborOp = [pos, rho0, h, &rho, &grad_pi_Ci, &sum_k_grad_Ci](std::size_t /*neighborId*/,
                                                                                     dtype *neighborPos,
                                                                                     dtype */*neighborVel*/) {
                // compute rho_i (equation 2)
                float len = util::distance<DIM>(pos, neighborPos);
                float tmp = util::Wpoly6(len, h);
                rho += tmp;

                // sum gradients of Ci (equation 8 and parts of equation 9)
                // use j as k so that we can stay in the same loop
                auto grad_pk_Ci = util::gradWspiky<DIM>(pos, neighborPos, h);

                for (auto i = 0u; i < DIM; ++i) {
                    grad_pk_Ci[i] /= rho0;
                }
                sum_k_grad_Ci += util::dot<DIM>(grad_pk_Ci.data(), grad_pk_Ci.data());

                for (auto i = 0u; i < DIM; ++i) {
                    grad_pi_Ci[i] += grad_pk_Ci[i];
                }
            };

            _neighborList.forEachNeighbor(id, _particles, neighborOp);
            sum_k_grad_Ci += util::dot<DIM>(grad_pi_Ci.data(), grad_pi_Ci.data());

            // compute lambda_i (equations 1 and 9)
            dtype C_i = rho / rho0 - 1;
            dtype lambda = -C_i / (sum_k_grad_Ci + _epsilon);
            lambdas.at(id) = lambda;
        };

        detail::forEachParticle(_particles, solverOp);
    }

    void updatePositions() {
        auto posmax = _neighborList.gridSize();
        auto posmin = posmax;
        for (auto i = 0u; i < DIM; ++i) {
            posmin[i] = -1 * 0.5 * posmax[i];
            posmax[i] *= 0.5;
        }
        auto updateOp = [this, posmin, posmax](std::size_t id, dtype *pos, dtype */*vel*/) {
            std::array<dtype, DIM> posDelta;
            std::fill(posDelta.begin(), posDelta.end(), 0);

            const auto &llambdas = this->lambdas;
            auto lambda = llambdas.at(id);
            auto tis = _tensileInstabilityScale;
            auto k = _tensileInstabilityK;
            auto h = _interactionRadius;

            auto neighborOp = [lambda, tis, k, h, pos, &llambdas, &posDelta](std::size_t nId, dtype *nPos, dtype */*nVel*/) {
                auto nLambda = llambdas.at(nId);

                dtype corr = tis * util::Wpoly6(util::distance<DIM>(pos, nPos), h);
                corr = -k * corr * corr * corr * corr;

                auto conv = util::gradWspiky<DIM>(pos, nPos, h);
                for (auto i = 0u; i < DIM; ++i) {
                    posDelta[i] += (lambda + nLambda + corr) * conv[i];
                }
            };

            _neighborList.forEachNeighbor(id, _particles, neighborOp);

            for (auto i = 0u; i < DIM; ++i) {
                pos[i] += posDelta[i] / _rho0;
                pos[i] = util::clamp(pos[i], posmin[i], posmax[i]);
            }
        };

        detail::forEachParticle(_particles, updateOp);
    }

    void update() {
        auto updateOp = [this](std::size_t id, dtype *pos, dtype *vel) {
            dtype *prevPos = _positions.data() + id * DIM;
            // update velocities and previous positions
            for (auto i = 0u; i < DIM; ++i) {
                vel[i] = (pos[i] - prevPos[i]) / _dt;
                prevPos[i] = pos[i];
            }
        };
        detail::forEachParticle(_particles, updateOp);
    }

    std::vector<dtype> run(std::uint32_t steps, dtype drift) {
        std::vector<dtype> trajectory;
        trajectory.reserve(steps * DIM * _particles.nParticles());
        auto backInserterIt = std::back_inserter(trajectory);
        for (auto step = 0U; step < steps; ++step) {
            predictPositions(drift);
            updateNeighborlist();
            for (auto i = 0u; i < _nSolverIterations; ++i) {
                calculateLambdas();
                updatePositions();
            }
            update();
            std::copy(_positions.begin(), _positions.end(), backInserterIt);
        }
        return trajectory;
    }

    std::size_t nParticles() const { return _particles.nParticles(); }

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

    auto gridSize() const { return _neighborList.gridSize(); }

    void setTensileInstabilityDistance(dtype deltaQ) {
        if(deltaQ >= _interactionRadius || deltaQ <= 0) {
            throw std::invalid_argument("Tensile instability distance should be positive but smaller than the "
                                        "interaction radius.");
        }
        _tensileInstabilityDistance = deltaQ;
        _tensileInstabilityScale = static_cast<dtype>(1. / util::Wpoly6(static_cast<dtype>(0.2), _interactionRadius));
    }

    dtype tensileInstabilityDistance() const { return _tensileInstabilityDistance; }

    void setTensileInstabilityK(dtype k) { _tensileInstabilityK = k; }

    dtype tensileInstabilityK() const { return _tensileInstabilityK; }

private:
    int nJobs{};
    dtype _interactionRadius{};
    std::vector<dtype> _positions{};
    std::vector<dtype> lambdas{};
    ParticleCollection<DIM, dtype> _particles{};
    NeighborList<DIM, dtype> _neighborList{};

    std::uint32_t _nSolverIterations = 5;
    dtype _gravity = static_cast<dtype>(10.);
    dtype _dt = static_cast<dtype>(0.016);
    dtype _rho0 = static_cast<dtype>(1.);
    dtype _epsilon = static_cast<dtype>(5.);
    dtype _tensileInstabilityDistance = static_cast<dtype>(0.2);
    dtype _tensileInstabilityScale = static_cast<dtype>(1. / util::Wpoly6(static_cast<dtype>(0.2), _interactionRadius));
    dtype _tensileInstabilityK = static_cast<dtype>(0.1);
};

}
