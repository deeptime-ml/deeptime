//
// Created by mho on 2/3/20.
//

#pragma once

#include <random>
#include <thread>
#include <iomanip>

#include "common.h"
#include "thread_utils.h"

namespace hmm {
namespace output_models {

template<typename dtype>
void handleOutliers(np_array_nfc<dtype> &outputProbabilityTrajectory) {
    auto nTimesteps = outputProbabilityTrajectory.shape(0);
    auto nStates = outputProbabilityTrajectory.shape(1);
    auto ptr = outputProbabilityTrajectory.mutable_data();

    #pragma omp parallel for
    for (decltype(nTimesteps) t = 0; t < nTimesteps; ++t) {
        auto begin = ptr + t * nStates;
        auto end = begin + nStates;
        auto sum = std::accumulate(begin, end, static_cast<dtype>(0.));
        if (sum == 0) {
            // got an outlier, fill with uniform (will be renormalized later)
            std::fill(begin, end, static_cast<dtype>(1));
        }
    }

}


namespace discrete {

template<typename dtype, typename State>
np_array<std::int64_t> generateObservationTrajectory(const np_array_nfc<State> &hiddenStateTrajectory,
                                                     const np_array_nfc<dtype> &outputProbabilities) {
    if (hiddenStateTrajectory.ndim() != 1) {
        throw std::invalid_argument("generate observation trajectory needs 1-dimensional hidden state trajectory");
    }
    auto nTimesteps = hiddenStateTrajectory.shape(0);
    auto nObs = outputProbabilities.shape(1);

    np_array<std::int64_t> output({nTimesteps});
    auto outputPtr = output.mutable_data();

    {
        auto nThreads = std::thread::hardware_concurrency();
        std::vector<deeptime::thread::scoped_thread> threads;
        threads.reserve(nThreads);
        auto grainSize = std::max(static_cast<pybind11::ssize_t>(1), nTimesteps / nThreads);

        const auto* hiddenStateTrajectoryBuf = hiddenStateTrajectory.data();
        const auto* outputProbabilitiesBuf = outputProbabilities.data();

        for(pybind11::ssize_t nextIndex = 0; nextIndex < nTimesteps; nextIndex += grainSize) {
            auto beginIndex = nextIndex;
            auto endIndex = std::min(nextIndex+grainSize, nTimesteps);
            threads.emplace_back([hiddenStateTrajectoryBuf, outputProbabilitiesBuf,
                                  beginIndex, endIndex, outputPtr, nObs]{
                auto generator = deeptime::rnd::randomlySeededGenerator();
                std::discrete_distribution<> ddist;
                for(pybind11::ssize_t t = beginIndex; t < endIndex; ++t) {
                    auto state = hiddenStateTrajectoryBuf[t];
                    auto begin = outputProbabilitiesBuf + state * nObs;  // outputProbabilities.at(state, 0)
                    auto end = outputProbabilitiesBuf + (state+1) * nObs;  // outputProbabilities.at(state+1, 0)
                    ddist.param(decltype(ddist)::param_type(begin, end));
                    auto obs = ddist(generator);
                    *(outputPtr + t) = obs;
                }
            });
        }
    }
    return output;
}

template<typename dtype, typename State>
np_array<dtype> toOutputProbabilityTrajectory(const np_array_nfc<State> &observations,
                                              const np_array_nfc<dtype> &outputProbabilities) {
    if (observations.ndim() != 1) {
        throw std::invalid_argument("observations trajectory needs to be one-dimensional.");
    }
    auto nHidden = static_cast<std::size_t>(outputProbabilities.shape(0));
    auto nObs = static_cast<std::size_t>(outputProbabilities.shape(1));
    const auto* P = outputProbabilities.data();
    const auto* obs = observations.data();

    np_array<dtype> output(std::vector<std::size_t>{static_cast<std::size_t>(observations.shape(0)), nHidden});
    auto* outputPtr = output.mutable_data();
    auto T = observations.shape(0);

    #pragma omp parallel for collapse(2) default(none) firstprivate(P, obs, nHidden, nObs, T, outputPtr)
    for (ssize_t t = 0; t < T; ++t) {
        for (std::size_t i = 0; i < nHidden; ++i) {
            outputPtr[t*nHidden + i] = P[obs[t] + i*nObs];
        }
    }

    return output;
}

template<typename dtype, typename State>
void sample(const std::vector<np_array_nfc<State>> &observationsPerState, np_array_nfc<dtype> &outputProbabilities,
            const np_array_nfc<dtype> &prior) {
    auto nObs = outputProbabilities.shape(1);
    ssize_t currentState{0};

    auto& generator = deeptime::rnd::staticThreadLocalGenerator();
    deeptime::rnd::dirichlet_distribution<dtype> dirichlet;

    for (const auto &observations : observationsPerState) {

        std::vector<dtype> hist(nObs, 0);

        #ifdef USE_OPENMP
        auto T = observations.size();
        auto* histPtr = hist.data();
        auto* observationsBuf = observations.data();

        #pragma omp parallel default(none) firstprivate(nObs, histPtr, T, observationsBuf)
        {
            std::vector<dtype> histPrivate(nObs, 0);

            #pragma omp for
            for(ssize_t i = 0; i < T; ++i) {
                ++histPrivate.at(observationsBuf[i]);
            }

            #pragma omp critical
            {
                for(int n = 0; n < nObs; ++n) {
                    histPtr[n] += histPrivate[n];
                }
            }
        }

        #else

        for (ssize_t i = 0; i < observations.size(); ++i) {
            ++hist.at(observations.at(i));
        }

        #endif // USE_OPENMP

        auto priorBegin = prior.data(currentState);
        // add prior onto histogram
        std::transform(hist.begin(), hist.end(), priorBegin, hist.begin(), std::plus<>());

        std::vector<std::size_t> positivesMapping;
        positivesMapping.reserve(hist.size());

        std::vector<dtype> reducedHist;
        reducedHist.reserve(hist.size());
        for(std::size_t i = 0; i < hist.size(); ++i) {
            if(hist.at(i) > 0) {
                positivesMapping.push_back(i);
                reducedHist.push_back(hist.at(i));
            }
        }

        dirichlet.params(reducedHist.begin(), reducedHist.end());
        auto probs = dirichlet(generator);

        for (std::size_t i = 0; i < probs.size(); ++i) {
            outputProbabilities.mutable_at(currentState, positivesMapping[i]) = probs[i];
        }

        ++currentState;
    }
}

template<typename dtype, typename State>
void updatePOut(const np_array_nfc<State> &obs, const np_array_nfc<dtype> &weights, np_array_nfc<dtype> &pout) {
    auto T = static_cast<std::size_t>(obs.size());
    auto N = static_cast<std::size_t>(pout.shape(0));
    auto M = static_cast<std::size_t>(pout.shape(1));

    auto obsBuf = obs.data();
    auto weightsBuf = weights.data();
    auto poutBuf = pout.mutable_data();

    for (std::size_t t = 0; t < T; ++t) {
        auto o = obsBuf[t];
        for (std::size_t i = 0; i < N; ++i) {
            poutBuf[i * M + o] += weightsBuf[t * N + i];
        }
    }
}

}

namespace gaussian {

/**
 * Returns the probability density of a Gaussian with given mu and sigma evaluated at o
 * @tparam dtype data type
 * @param o observation value
 * @param mu mean value
 * @param sigma standard deviation
 */
template<typename dtype>
constexpr dtype sample(dtype o, dtype mu, dtype sigma) {
    #ifndef _WIN32
    double c = 1.0 / (std::sqrt(2.0 * dt::constants::pi<dtype>()) * sigma);
    double d = (o - mu) / sigma;
    return c * std::exp(-0.5 * d * d);
    #else
    return exp(-0.5 * ((o - mu) / sigma) * ((o - mu) / sigma)) / (std::sqrt(2.0 * dt::constants::pi<dtype>()) * sigma);
    #endif
}

template<typename dtype>
np_array<dtype> pO(dtype o, const np_array_nfc<dtype> &mus, const np_array_nfc<dtype> &sigmas, py::object out) {
    auto N = mus.shape(0);

    np_array<dtype> p;
    if (!out.is_none()) {
        p = py::cast<np_array<dtype>>(out);
    } else {
        p = np_array<dtype>({N});
    }
    auto pBuf = p.mutable_data();
    auto musBuf = mus.data();
    auto sigmasBuf = sigmas.data();

    #pragma omp parallel for
    for (pybind11::ssize_t i = 0; i < N; ++i) {
        pBuf[i] = sample(o, musBuf[i], sigmasBuf[i]);
    }

    return p;
}

template<typename dtype>
np_array<dtype> toOutputProbabilityTrajectory(const np_array_nfc<dtype> &obs, const np_array_nfc<dtype> &mus,
                                              const np_array_nfc<dtype> &sigmas) {
    auto N = static_cast<std::size_t>(mus.shape(0));
    auto T = static_cast<std::size_t>(obs.shape(0));

    np_array<dtype> p({T, N});
    auto obsBuf = obs.data();
    auto musBuf = mus.data();
    auto sigmasBuf = sigmas.data();
    auto pBuf = p.mutable_data();

    #pragma omp parallel for collapse(2)
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t i = 0; i < N; ++i) {
            pBuf[t * N + i] = sample(obsBuf[t], musBuf[i], sigmasBuf[i]);
        }
    }

    return p;
}

template<typename dtype>
np_array<dtype>
generateObservationTrajectory(const np_array_nfc<dtype> &hiddenStateTrajectory, const np_array_nfc<dtype> &means,
                              const np_array_nfc<dtype> &sigmas) {
    if (hiddenStateTrajectory.ndim() != 1) {
        throw std::invalid_argument("Hidden state trajectory must be one-dimensional!");
    }
    auto nTimesteps = hiddenStateTrajectory.shape(0);
    np_array<dtype> output({nTimesteps});
    auto ptr = output.mutable_data();

    std::normal_distribution<dtype> dist{0, 1};
    for (decltype(nTimesteps) t = 0; t < nTimesteps; ++t) {
        auto state = hiddenStateTrajectory.at(t);
        *(ptr + t) = sigmas.at(state) * dist(deeptime::rnd::staticThreadLocalGenerator()) + means.at(state);
    }
    return output;
}

template<typename dtype>
std::tuple<np_array<dtype>, np_array<dtype>> fit(std::size_t nHiddenStates, const py::list &observations,
                                                 const py::list &weights) {
    auto nObsTrajs = observations.size();
    if (nObsTrajs != weights.size()) {
        throw std::invalid_argument("number of observation trajectories must match number of weight matrices");
    }

    auto result = std::make_tuple(
            np_array<dtype>(std::vector<std::size_t>{nHiddenStates}),
            np_array<dtype>(std::vector<std::size_t>{nHiddenStates})
    );
    auto &means = std::get<0>(result);
    auto &sigmas = std::get<1>(result);

    std::fill(means.mutable_data(), means.mutable_data() + nHiddenStates, 0);
    std::fill(sigmas.mutable_data(), sigmas.mutable_data() + nHiddenStates, 0);

    // fit means
    {
        std::vector<dtype> wSum(nHiddenStates, 0);
        auto weightsIt = weights.begin();
        auto obsIt = observations.begin();
        for (decltype(nObsTrajs) k = 0; k < nObsTrajs; ++k, ++weightsIt, ++obsIt) {
            const auto &w = py::cast<np_array<dtype>>(*weightsIt);
            const auto &obs = py::cast<np_array<dtype>>(*obsIt);
            const auto* obsPtr = obs.data();
            for (decltype(nHiddenStates) i = 0; i < nHiddenStates; ++i) {
                dtype dot = 0;
                dtype wStateSum = 0;
                for (ssize_t t = 0; t < obs.shape(0); ++t) {
                    dot += w.at(t, i) * obsPtr[t];
                    wStateSum += w.at(t, i);
                }
                // update nominator
                means.mutable_at(i) += dot;
                // update denominator
                wSum.at(i) += wStateSum;
            }

        }
        // update normalize
        for(decltype(nHiddenStates) i = 0; i < nHiddenStates; ++i) {
            means.mutable_at(i) /= wSum.at(i);
        }
    }
    // fit variances
    {
        std::vector<dtype> wSum(nHiddenStates, 0);
        auto weightsIt = weights.begin();
        auto obsIt = observations.begin();
        for (decltype(nObsTrajs) k = 0; k < nObsTrajs; ++k, ++weightsIt, ++obsIt) {
            const auto &w = py::cast<np_array<dtype>>(*weightsIt);
            const auto &obs = py::cast<np_array<dtype>>(*obsIt);
            const auto *obsPtr = obs.data();

            for (decltype(nHiddenStates) i = 0; i < nHiddenStates; ++i) {
                dtype wStateSum = 0;
                dtype sigmaUpdate = 0;
                for (ssize_t t = 0; t < obs.shape(0); ++t) {
                    auto sqrty = static_cast<dtype>(obsPtr[t]) - static_cast<dtype>(means.at(i));
                    sigmaUpdate += w.at(t, i) * sqrty*sqrty;
                    wStateSum += w.at(t, i);
                }
                // update nominator
                sigmas.mutable_at(i) += sigmaUpdate;
                // update denominator
                wSum.at(i) += wStateSum;
            }
        }
        for(decltype(nHiddenStates) i = 0; i < nHiddenStates; ++i) {
            sigmas.mutable_at(i) = std::sqrt(sigmas.at(i) / wSum.at(i));
        }
    }
    return result;
}

}

}
}
