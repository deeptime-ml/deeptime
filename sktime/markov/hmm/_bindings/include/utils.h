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

    auto alphaPtr = alpha.mutable_data();
    auto transitionMatrixPtr = transitionMatrix.data();
    auto pObsPtr = pObs.data();
    auto piPtr = pi.data();

    auto N = static_cast<std::size_t>(transitionMatrix.shape(0));

    dtype sum, logprob, scaling;
    {
        py::gil_scoped_release gil;

        // first alpha and scaling factors
        scaling = 0.0;
        for (std::size_t i = 0; i < N; i++) {
            alphaPtr[i] = piPtr[i] * pObsPtr[i];
            scaling += alphaPtr[i];
        }

        // initialize likelihood
        logprob = std::log(scaling);

        // scale first alpha
        if (scaling != 0) {
            for (std::size_t i = 0; i < N; i++) {
                alphaPtr[i] /= scaling;
            }
        }

        // iterate trajectory
        for (std::size_t t = 0; t < T - 1; t++) {
            scaling = 0.0;
            // compute new alpha and scaling
            for (std::size_t j = 0; j < N; j++) {
                sum = 0.0;
                for (std::size_t i = 0; i < N; i++) {
                    sum += alphaPtr[t * N + i] * transitionMatrixPtr[i * N + j];
                }
                alphaPtr[(t + 1) * N + j] = sum * pObsPtr[(t + 1) * N + j];
                scaling += alphaPtr[(t + 1) * N + j];
            }
            // scale this row
            if (scaling != 0) {
                for (std::size_t j = 0; j < N; j++) {
                    alphaPtr[(t + 1) * N + j] /= scaling;
                }
            }

            // update likelihood
            logprob += std::log(scaling);
        }
    }

    return logprob;
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
        throw std::invalid_argument("Shape mismatch: Beta must have at least size T and otherwise shape of beta must "
                                    "match shape of state probability trajectory.");
    }

    auto transitionMatrixPtr = transitionMatrix.data();
    auto pObsPtr = pobs.data();
    auto betaPtr = beta.mutable_data();

    auto N = static_cast<std::size_t>(transitionMatrix.shape(0));
    {
        py::gil_scoped_release gil;

        std::size_t i, j, t;
        dtype sum, scaling;

        // first beta and scaling factors
        scaling = 0.0;
        for (i = 0; i < N; ++i) {
            betaPtr[(T - 1) * N + i] = 1.0;
            scaling += betaPtr[(T - 1) * N + i];
        }

        // scale first beta
        for (i = 0; i < N; ++i) {
            betaPtr[(T - 1) * N + i] /= scaling;
        }

        // iterate trajectory
        for (t = T - 1; t >= 1; --t) {
            scaling = 0.0;
            // compute new beta and scaling
            for (i = 0; i < N; ++i) {
                sum = 0.0;
                for (j = 0; j < N; ++j) {
                    sum += transitionMatrixPtr[i * N + j] * pObsPtr[t * N + j] * betaPtr[t * N + j];
                }
                betaPtr[(t - 1) * N + i] = sum;
                scaling += sum;
            }
            // scale this row
            if (scaling != 0) {
                for (j = 0; j < N; ++j) {
                    betaPtr[(t - 1) * N + j] /= scaling;
                }
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

    #pragma omp parallel for
    for (std::size_t t = 0; t < T; ++t) {
        dtype rowSum = 0;
        for (std::size_t n = 0; n < N; ++n) {
            *(gammaPtr + t * N + n) = *(alphaPtr + t * N + n) * *(betaPtr + t * N + n);
            rowSum += gammaPtr[t * N + n];
        }
        for (std::size_t n = 0; n < N; ++n) {
            if (rowSum != 0.) {
                gammaPtr[t * N + n] /= rowSum;
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

    py::gil_scoped_release gil;

    std::fill(countsBuf, countsBuf + counts.size(), 0.0);

    dtype sum;

    std::unique_ptr<dtype[]> tmp = std::unique_ptr<dtype[]>(new dtype[N * N]);

    for (std::size_t t = 0; t < T - 1; t++) {
        sum = 0.0;
        for (std::size_t i = 0; i < N; i++) {
            for (std::size_t j = 0; j < N; j++) {
                tmp[i * N + j] =
                        alphaBuf[t * N + i] * transitionMatrixPtr[i * N + j] * pObsBuf[(t + 1) * N + j] *
                        betaBuf[(t + 1) * N + j];
                sum += tmp[i * N + j];
            }
        }
        for (std::size_t i = 0; i < N; i++) {
            for (std::size_t j = 0; j < N; j++) {
                countsBuf[i * N + j] += tmp[i * N + j] / sum;
            }
        }
    }
}

template<typename dtype>
np_array<std::int32_t>
samplePath(const np_array<dtype> &alpha, const np_array<dtype> &transitionMatrix, std::size_t T, int seed = -1) {
    auto N = static_cast<std::size_t>(transitionMatrix.shape(0));

    np_array<std::int32_t> pathArray(std::vector<std::size_t>{T});
    auto path = pathArray.mutable_data();

    auto pselPtr = std::unique_ptr<dtype[]>(new dtype[N]);
    auto psel = pselPtr.get();

    //auto alphaBuf = alpha.data();
    //auto transitionMatrixPtr = transitionMatrix.data();
    {
        // py::gil_scoped_release gil;
        //std::default_random_engine generator(seed >= 0 ? seed : clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
        auto &generator = sktime::rnd::staticGenerator();

        // Sample final state.
        for (std::size_t i = 0; i < N; i++) {
            psel[i] = alpha.at(T-1, i); // [(T - 1) * N + i];
        }
        //normalize(psel, psel + N);

        std::discrete_distribution<> ddist(psel, psel + N);
        // Draw from this distribution.
        path[T - 1] = ddist(generator); //_random_choice(psel, N);

        // Work backwards from T-2 to 0.
        for (std::int32_t t = T - 2; t >= 0; --t) {
            // Compute P(s_t = i | s_{t+1}..s_T).
            for (std::size_t i = 0; i < N; ++i) {
                psel[i] = alpha.at(t, i) * transitionMatrix.at(i, path[t+1]); // alphaBuf[(t - 1) * N + i] * transitionMatrixPtr[i * N + path[t]];
            }
            //normalize(psel, psel + N);
            ddist.param(decltype(ddist)::param_type(psel, psel + N));

            // Draw from this distribution.
            path[t] = ddist(generator); //_random_choice(psel, N);
        }
    }

    return pathArray;
}
