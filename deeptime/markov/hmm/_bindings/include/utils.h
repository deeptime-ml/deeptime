//
// Created by mho on 2/10/20.
//

#pragma once

#include <thread>

#include "common.h"
#include "distribution_utils.h"

/**
 * computes viterbi path
 * @tparam dtype dtype
 * @param transitionMatrix (N, N) transition matrix
 * @param stateProbabilityTraj (T, N) pobs
 * @param initialDistribution (N,) init dist
 * @return (T, )ndarray
 */
template<typename dtype>
np_array<std::int32_t> viterbiPath(const np_array<dtype> &transitionMatrix, const np_array<dtype> &stateProbabilityTraj,
                                   const np_array<dtype> &initialDistribution) {
    if (transitionMatrix.ndim() < 1 || stateProbabilityTraj.ndim() < 1) {
        throw std::invalid_argument("transition matrix and pobs need to be at least 1-dimensional.");
    }
    auto N = static_cast<std::size_t>(transitionMatrix.shape(0));
    auto T = static_cast<std::size_t>(stateProbabilityTraj.shape(0));
    {
        // check shapes
        if (transitionMatrix.ndim() != 2) throw std::invalid_argument("transition matrix must be 2-dimensional");
        if (transitionMatrix.shape(1) != transitionMatrix.shape(0)) {
            throw std::invalid_argument("Transition matrix must be (N, N) but was (" + std::to_string(N) + ", " +
                                        std::to_string(transitionMatrix.shape(1)) + ")");
        }
        if (stateProbabilityTraj.ndim() != 2) throw std::invalid_argument("pobs must be 2-dimensional");
        if (static_cast<std::size_t>(stateProbabilityTraj.shape(1)) != N) {
            std::stringstream ss;
            ss << "State probablity trajectory must be (T, N) = (" << T << ", " << N << ") dimensional but was (";
            ss << stateProbabilityTraj.shape(0) << ", " << stateProbabilityTraj.shape(1) << ")";
            throw std::invalid_argument(ss.str());
        }
        if (initialDistribution.ndim() != 1) throw std::invalid_argument("initial distribution must be 1-dimensional");
        if (static_cast<std::size_t>(initialDistribution.shape(0)) != N) {
            throw std::invalid_argument(
                    "initial distribution must have length N = " + std::to_string(N) + " but had len=" +
                    std::to_string(initialDistribution.shape(0)));
        }
        if (T == 0 || N == 0) {
            throw std::invalid_argument("Needs T and N to be at least 1, i.e., no empty arrays permitted.");
        }
    }
    np_array<std::int32_t> path(std::vector<std::size_t>{T});
    auto pathBuf = path.mutable_data();
    auto ABuf = transitionMatrix.data();
    auto pobsBuf = stateProbabilityTraj.data();
    auto piBuf = initialDistribution.data();
    {
        std::fill(pathBuf, pathBuf + T, 0);

        dtype sum;
        auto vData = std::unique_ptr<dtype[]>(new dtype[N]);
        auto v = vData.get();
        auto vnextData = std::unique_ptr<dtype[]>(new dtype[N]);
        auto vnext = vnextData.get();
        auto hData = std::unique_ptr<dtype[]>(new dtype[N]);
        auto h = hData.get();
        auto pathTmpBuf = std::unique_ptr<std::int32_t[]>(new std::int32_t[T * N]);
        auto ptr = pathTmpBuf.get();

        // initialization of v
        sum = 0.0;
        for (std::size_t i = 0; i < N; i++) {
            v[i] = pobsBuf[i] * piBuf[i];
            sum += v[i];
        }
        // normalize
        for (std::size_t i = 0; i < N; i++) {
            v[i] /= sum;
        }

        // iteration of v
        for (std::size_t t = 1; t < T; t++) {
            sum = 0.0;
            for (std::size_t j = 0; j < N; j++) {
                for (std::size_t i = 0; i < N; i++) {
                    h[i] = v[i] * ABuf[i * N + j];
                }
                auto maxi = std::distance(h, std::max_element(h, h + N));
                ptr[t * N + j] = maxi;
                vnext[j] = pobsBuf[t * N + j] * v[maxi] * ABuf[maxi * N + j];
                sum += vnext[j];
            }
            // normalize
            for (std::size_t i = 0; i < N; i++) {
                vnext[i] /= sum;
            }
            // update v
            std::swap(v, vnext);
        }

        // path reconstruction
        pathBuf[T - 1] = std::distance(v, std::max_element(v, v + N));
        for (std::size_t t = T - 1; t >= 1; t--) {
            pathBuf[t - 1] = ptr[t * N + pathBuf[t]];
        }


    }
    return path;
}

template<typename dtype>
dtype forwardImpl(const dtype*const  transitionMatrix, const dtype*const  pObs, const dtype*const pi,
                  dtype* const alpha, std::size_t N, std::size_t T) {

    dtype logprob, scaling;
    {
        // first alpha and scaling factors
        scaling = 0.0;
        //#pragma omp parallel for reduction(+:scaling) default(none) firstprivate(N, alpha, pi, pObs)
        for (std::size_t i = 0; i < N; i++) {
            alpha[i] = pi[i] * pObs[i];
            scaling += alpha[i];
        }

        // initialize likelihood
        logprob = std::log(scaling);

        // scale first alpha
        if (scaling != 0) {
            //#pragma omp parallel for default(none) firstprivate(N, alpha, scaling)
            for (std::size_t i = 0; i < N; i++) {
                alpha[i] /= scaling;
            }
        }

        // iterate trajectory
        for (std::size_t t = 0; t < T - 1; t++) {
            scaling = 0.0;
            // compute new alpha and scaling
            //#pragma omp parallel for reduction(+:scaling) default(none) firstprivate(N, alpha, transitionMatrix, pObs, t)
            for (std::size_t j = 0; j < N; j++) {
                dtype sum = 0.0;
                for (std::size_t i = 0; i < N; i++) {
                    sum += alpha[t * N + i] * transitionMatrix[i * N + j];
                }
                alpha[(t + 1) * N + j] = sum * pObs[(t + 1) * N + j];
                scaling += alpha[(t + 1) * N + j];
            }
            // scale this row
            if (scaling != 0) {
                //#pragma omp parallel for default(none) firstprivate(N, alpha, scaling, t)
                for (std::size_t j = 0; j < N; j++) {
                    alpha[(t + 1) * N + j] /= scaling;
                }
            }

            // update likelihood
            logprob += std::log(scaling);
        }
    }

    return logprob;
}

template<typename dtype>
dtype forward(const np_array<dtype> &transitionMatrix, const np_array<dtype> &pObs, const np_array<dtype> &pi,
              np_array<dtype> &alpha, const py::object &pyT) {
    std::size_t T = [&pyT, &pObs]() {
        if (pyT.is_none()) {
            return static_cast<std::size_t>(pObs.shape(0));
        } else {
            return py::cast<std::size_t>(pyT);
        }
    }();

    if (T > static_cast<std::size_t>(pObs.shape(0))) {
        throw std::invalid_argument("T must be at most the length of pobs.");
    }
    if (alpha.ndim() != pObs.ndim() || static_cast<std::size_t>(alpha.shape(0)) < T ||
        alpha.shape(1) != pObs.shape(1)) {
        throw std::invalid_argument("Shape mismatch: Shape of state probability trajectory must match shape of alphas");
    }
    auto N = static_cast<std::size_t>(transitionMatrix.shape(0));
    return forwardImpl(transitionMatrix.data(), pObs.data(), pi.data(), alpha.mutable_data(), N, T);
}

template<typename dtype>
void backwardImpl(const dtype* const transitionMatrix, const dtype* const pobs, dtype* const beta, std::size_t N,
                  std::size_t T) {
    {
        std::size_t i, j, t;
        dtype sum, scaling;

        // first beta and scaling factors
        scaling = 0.0;
        for (i = 0; i < N; ++i) {
            beta[(T - 1) * N + i] = 1.0;
            scaling += beta[(T - 1) * N + i];
        }

        // scale first beta
        for (i = 0; i < N; ++i) {
            beta[(T - 1) * N + i] /= scaling;
        }

        // iterate trajectory
        for (t = T - 1; t >= 1; --t) {
            scaling = 0.0;
            // compute new beta and scaling
            for (i = 0; i < N; ++i) {
                sum = 0.0;
                for (j = 0; j < N; ++j) {
                    sum += transitionMatrix[i * N + j] * pobs[t * N + j] * beta[t * N + j];
                }
                beta[(t - 1) * N + i] = sum;
                scaling += sum;
            }
            // scale this row
            if (scaling != 0) {
                for (j = 0; j < N; ++j) {
                    beta[(t - 1) * N + j] /= scaling;
                }
            }
        }
    }
}

template<typename dtype>
void backward(const np_array<dtype> &transitionMatrix, const np_array<dtype> &pobs, np_array<dtype> &beta,
              const py::object &pyT) {
    std::size_t T = [&pyT, &pobs]() {
        if (pyT.is_none()) {
            return static_cast<std::size_t>(pobs.shape(0));
        } else {
            return py::cast<std::size_t>(pyT);
        }
    }();

    if (beta.ndim() != pobs.ndim() || static_cast<std::size_t>(beta.shape(0)) < T || beta.shape(1) != pobs.shape(1)) {
        throw std::invalid_argument("Shape mismatch: Beta must have at least length T and the length of the second "
                                    "dimension of beta must match that of the state probability trajectory.");
    }

    auto N = static_cast<std::size_t>(transitionMatrix.shape(0));

    backwardImpl(transitionMatrix.data(), pobs.data(), beta.mutable_data(), N, T);
}

template<typename dtype>
void stateProbabilitiesImpl(const dtype* const alpha, const dtype* const beta, dtype* const gamma,
                            std::size_t N, std::size_t T) {
    #pragma omp parallel for default(none) firstprivate(T, N, alpha, beta, gamma)
    for (std::size_t t = 0; t < T; ++t) {
        dtype rowSum = 0;
        for (std::size_t n = 0; n < N; ++n) {
            *(gamma + t * N + n) = *(alpha + t * N + n) * *(beta + t * N + n);
            rowSum += gamma[t * N + n];
        }
        for (std::size_t n = 0; n < N; ++n) {
            if (rowSum != 0.) {
                gamma[t * N + n] /= rowSum;
            }
        }
    }
}

template<typename dtype>
void stateProbabilities(const np_array<dtype> &alpha, const np_array<dtype> &beta, np_array<dtype> &gamma,
                        const py::object &pyT) {
    std::size_t T = [&pyT, &gamma]() {
        if (pyT.is_none()) {
            return static_cast<std::size_t>(gamma.shape(0));
        } else {
            return py::cast<std::size_t>(pyT);
        }
    }();
    auto N = static_cast<std::size_t>(alpha.shape(1));
    auto alphaPtr = alpha.data();
    auto betaPtr = beta.data();
    auto gammaPtr = gamma.mutable_data();
    stateProbabilitiesImpl(alphaPtr, betaPtr, gammaPtr, N, T);
}

template<typename dtype>
void transitionCountsImpl(const dtype* const alpha, const dtype* const beta, const dtype* const transitionMatrix,
                          const dtype* const pObs, dtype* const counts, std::size_t N, std::size_t T) {
    std::fill(counts, counts + N*N, 0.0);

    dtype sum;

    std::unique_ptr<dtype[]> tmp = std::unique_ptr<dtype[]>(new dtype[N * N]);

    for (std::size_t t = 0; t < T - 1; t++) {
        sum = 0.0;
        for (std::size_t i = 0; i < N; i++) {
            for (std::size_t j = 0; j < N; j++) {
                tmp[i * N + j] =
                        alpha[t * N + i] * transitionMatrix[i * N + j] * pObs[(t + 1) * N + j] *
                        beta[(t + 1) * N + j];
                sum += tmp[i * N + j];
            }
        }
        for (std::size_t i = 0; i < N; i++) {
            for (std::size_t j = 0; j < N; j++) {
                counts[i * N + j] += tmp[i * N + j] / sum;
            }
        }
    }
}

template<typename dtype>
void transitionCounts(const np_array<dtype> &alpha, const np_array<dtype> &beta,
                      const np_array<dtype> &transitionMatrix,
                      const np_array<dtype> &pObs, np_array<dtype> &counts, const py::object &pyT) {
    std::size_t T = [&pyT, &pObs]() {
        if (pyT.is_none()) {
            return static_cast<std::size_t>(pObs.shape(0));
        } else {
            return py::cast<std::size_t>(pyT);
        }
    }();
    if (static_cast<std::size_t>(pObs.shape(0)) < T) {
        throw std::invalid_argument("T must be at least the length of pObs.");
    }
    if (!arraySameShape(transitionMatrix, counts)) {
        throw std::invalid_argument("Shape mismatch: counts must be same shape as transition matrix.");
    }
    auto countsBuf = counts.mutable_data();
    auto alphaBuf = alpha.data();
    auto transitionMatrixPtr = transitionMatrix.data();
    auto betaBuf = beta.data();
    auto pObsBuf = pObs.data();

    auto N = static_cast<std::size_t>(transitionMatrix.shape(0));

    transitionCountsImpl(alphaBuf, betaBuf, transitionMatrixPtr, pObsBuf, countsBuf, N, T);
}

template<typename dtype, typename Generator>
void samplePathImpl(const dtype* const alpha, const dtype* const transitionMatrix, std::size_t N, std::size_t T,
                    Generator& generator, std::int32_t* path) {
    auto pselPtr = std::unique_ptr<dtype[]>(new dtype[N]);
    auto psel = pselPtr.get();

    // Sample final state.
    for (std::size_t i = 0; i < N; i++) {
        psel[i] = alpha[(T-1)*N + i];
        if (i > 0) {
            psel[i] += psel[i-1];
        }
    }

    auto sample = [&generator, N, psel]() {
        static thread_local std::uniform_real_distribution<dtype> uiDist (0, 1);
        auto p = uiDist(generator) * psel[N - 1];  // uniform between 0 and psel[N-1] which is cumulative
        auto it = std::lower_bound(psel, psel + N, p);
        return std::distance(psel, it);
    };

    // Draw from this distribution.
    path[T - 1] = sample();

    // Work backwards from T-2 to 0.
    for (std::int64_t t = T - 2; t >= 0; --t) {
        // Compute P(s_t = i | s_{t+1}..s_T).
        for (std::size_t i = 0; i < N; ++i) {
            psel[i] = alpha[t*N + i] * transitionMatrix[i*N + path[t+1]];
            if (i > 0) {
                psel[i] += psel[i - 1];
            }
        }

        // Draw from this distribution.
        path[t] = sample();
    }
}

template<typename dtype>
np_array<std::int32_t>
samplePath(const np_array<dtype> &alpha, const np_array<dtype> &transitionMatrix, std::size_t T, int seed = -1) {
    auto N = static_cast<std::size_t>(transitionMatrix.shape(0));

    np_array<std::int32_t> pathArray(std::vector<std::size_t>{T});
    auto path = pathArray.mutable_data();

    if (seed < 0) {
        samplePathImpl(alpha.data(), transitionMatrix.data(), N, T, deeptime::rnd::staticThreadLocalGenerator(), path);
    } else {
        auto generator = deeptime::rnd::seededGenerator(static_cast<std::uint32_t>(seed));
        samplePathImpl(alpha.data(), transitionMatrix.data(), N, T, generator, path);
    }
    return pathArray;
}

template<typename dtype>
np_array<dtype> countMatrix(const py::list& dtrajs, std::uint32_t lag, std::uint32_t nStates) {
    np_array<dtype> result ({nStates, nStates});
    auto* buf = result.mutable_data();
    std::fill(buf, buf + nStates * nStates, static_cast<dtype>(0));

    for(auto traj : dtrajs) {
        auto npTraj = traj.cast<np_array<dtype>>();

        auto T = npTraj.shape(0);

        if (T > lag) {
            for(std::size_t t = 0; t < static_cast<std::size_t>(T - lag); ++t) {
                auto state1 = npTraj.at(t);
                auto state2 = npTraj.at(t+lag);
                ++buf[nStates * state1 + state2];
            }
        }
    }

    return result;
}

template<typename dtype>
dtype forwardBackward(const np_array<dtype> &transitionMatrix, const np_array<dtype> &pObs,
    const np_array<dtype> &pi, np_array<dtype> &alpha, np_array_nfc<dtype> &beta, np_array_nfc<dtype> &gamma,
    np_array_nfc<dtype> &counts, const py::object &pyT) {
    std::size_t T = [&pyT, &pObs]() {
        if (pyT.is_none()) {
            return static_cast<std::size_t>(pObs.shape(0));
        } else {
            return py::cast<std::size_t>(pyT);
        }
    }();
    if (T > static_cast<std::size_t>(pObs.shape(0))) {
        throw std::invalid_argument("T must be at most the length of pobs.");
    }
    if (alpha.ndim() != pObs.ndim() || static_cast<std::size_t>(alpha.shape(0)) < T ||
        alpha.shape(1) != pObs.shape(1)) {
        throw std::invalid_argument("Shape mismatch: Shape of state probability trajectory must match shape of alphas");
    }

    auto N = static_cast<std::size_t>(transitionMatrix.shape(0));

    const auto* P = transitionMatrix.data();
    const auto* pObsBuf = pObs.data();
    const auto* piBuf = pi.data();

    auto* alphaBuf = alpha.mutable_data();
    auto* betaBuf = beta.mutable_data();
    auto* gammaBuf = gamma.mutable_data();
    auto* countsBuf = counts.mutable_data();

    auto logprob = forwardImpl(P, pObsBuf, piBuf, alphaBuf, N, T);
    backwardImpl(P, pObsBuf, betaBuf, N, T);
    stateProbabilitiesImpl(alphaBuf, betaBuf, gammaBuf, N, T);
    transitionCountsImpl(alphaBuf, betaBuf, P, pObsBuf, countsBuf, N, T);
    return logprob;
}
