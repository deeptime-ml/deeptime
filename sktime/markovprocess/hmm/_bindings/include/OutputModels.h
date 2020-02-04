//
// Created by mho on 2/3/20.
//

#pragma once

#include <random>
#include <thread>

namespace hmm {
namespace output_models {

template<typename RealType>
class dirichlet_distribution {
public:
    dirichlet_distribution() : gammas() {}
    template<typename InputIterator>
    dirichlet_distribution(InputIterator wbegin, InputIterator wend) {
        params(wbegin, wend);
    }

    template<typename Generator>
    std::vector<RealType> operator()(Generator& gen) {
        std::vector<RealType> xs;
        xs.reserve(gammas.size());
        for(auto& gdist : gammas) {
            // ignore zeros
            if(gdist.alpha() != 0) {
                xs.push_back(gdist(gen));
            } else {
                xs.push_back(0);
            }
        }
        auto sum = std::accumulate(xs.begin(), xs.end(), 0);
        for(auto it = xs.begin(); it != xs.end(); ++it) {
            *it /= sum;
        }
        return xs;
    }

    template<typename InputIterator>
    void params(InputIterator wbegin, InputIterator wend) {
        gammas.resize(0);
        std::transform(wbegin, wend, std::back_inserter(gammas), [](const auto& weight) {
            return std::gamma_distribution<RealType>(weight, 1);
        });
    }
private:
    std::vector<std::gamma_distribution<RealType>> gammas;
};

template<template<typename, typename> class T, typename PROB, typename STATE>
struct OutputModel {
    using Subclass = T<PROB, STATE>;
    OutputModel() : _nHiddenStates(0), _nObservableStates(0), _ignoreOutliers(true) {}
    OutputModel(std::size_t nHiddenStates, std::size_t nObservableStates, bool ignoreOutliers)
            : _nHiddenStates(nHiddenStates), _nObservableStates(nObservableStates), _ignoreOutliers(ignoreOutliers) {
    }
    OutputModel(const OutputModel&) = default;
    OutputModel& operator=(const OutputModel&) = default;

    virtual ~OutputModel() = default;

    std::size_t nHiddenStates() const { return _nHiddenStates; }
    void setNHiddenStates(std::size_t value) { _nHiddenStates = value; }

    std::size_t nObservableStates() const { return _nObservableStates; }
    void setNObservableStates(std::size_t value) { _nObservableStates = value; }

    bool ignoreOutliers() const { return _ignoreOutliers; }
    void setIgnoreOutliers(bool value) { _ignoreOutliers = value; }

    Subclass submodel(const np_array<STATE>& states) const {
        return static_cast<const Subclass*>(this)->submodelImpl(states);
    }

    np_array<PROB> outputProbabilityTrajectory(const np_array<STATE>& observations) const {
        return static_cast<const Subclass*>(this)->outputProbabilityTrajectoryImpl(observations);
    }

    void handleOutliers(np_array<PROB> &outputProbabilityTrajectory) const {
        auto nTimesteps = outputProbabilityTrajectory.shape(0);
        auto nStates = outputProbabilityTrajectory.shape(1);
        auto ptr = outputProbabilityTrajectory.mutable_data();

        #pragma omp parallel for
        for(decltype(nTimesteps) t = 0; t < nTimesteps; ++t) {
            PROB sum = 0;
            for(decltype(nStates) i = 0; i < nStates; ++i) {
                sum += *(ptr + t * nStates + i);
            }
            if(sum == 0) {
                // got an outlier
                for(decltype(nStates) i = 0; i < nStates; ++i) {
                    std::fill(ptr + t*nStates, ptr + (t+1)*nStates, static_cast<PROB>(1));
                }
            }
        }

    }

    np_array<PROB> generateObservationTrajectory(const np_array<STATE> &hiddenStateTrajectory) const {
        return static_cast<const Subclass*>(this)->generateObservationTrajectoryImpl(hiddenStateTrajectory);
    }

    virtual np_array<PROB> generateObservationTrajectoryImpl(const np_array<STATE> &hiddenStateTrajectory) const = 0;
    virtual np_array<PROB> outputProbabilityTrajectoryImpl(const np_array<STATE>& observations) const = 0;
    virtual Subclass submodelImpl(const np_array<STATE>& states) const = 0;

protected:
    std::size_t _nHiddenStates;
    std::size_t _nObservableStates;
    bool _ignoreOutliers;
};

namespace gaussian {


template<typename dtype>
constexpr dtype pi() { return 3.141592653589793238462643383279502884e+00; }

/**
 * Returns the probability density of a Gaussian with given mu and sigma evaluated at o
 * @tparam dtype data type
 * @param o observation value
 * @param mu mean value
 * @param sigma standard deviation
 */
template<typename dtype>
constexpr dtype sample(dtype o, dtype mu, dtype sigma) {
    double c = 1.0 / (std::sqrt(2.0 * pi<dtype>()) * sigma);
    double d = (o - mu) / sigma;
    return c * exp(-0.5 * d * d);
}

template<typename dtype>
np_array<dtype> pO(dtype o, const np_array<dtype> &mus, const np_array<dtype> &sigmas, py::object out) {
    auto N = static_cast<std::size_t>(mus.shape(0));

    np_array<dtype> p;
    if(!out.is_none()) {
        p = py::cast<np_array<dtype>>(out);
    } else {
        p = np_array<dtype>({N});
    }
    auto pBuf = p.mutable_data();
    auto musBuf = mus.data();
    auto sigmasBuf = sigmas.data();

#pragma omp parallel for
    for(std::size_t i = 0; i < N; ++i) {
        pBuf[i] = sample(o, musBuf[i], sigmasBuf[i]);
    }

    return p;
}

template<typename dtype>
np_array<dtype> pObs(const np_array<dtype> &obs, const np_array<dtype> &mus, const np_array<dtype> &sigmas, py::object out) {
    auto N = static_cast<std::size_t>(mus.shape(0));
    auto T = static_cast<std::size_t>(obs.shape(0));

    np_array<dtype> p;
    if(!out.is_none()) {
        p = py::cast<np_array<dtype>>(out);
    } else {
        p = np_array<dtype>({T, N});
    }
    auto obsBuf = obs.data();
    auto musBuf = mus.data();
    auto sigmasBuf = sigmas.data();
    auto pBuf = p.mutable_data();

#pragma omp parallel for collapse(2)
    for (std::size_t t=0; t<T; ++t) {
        for (std::size_t i = 0; i < N; ++i) {
            pBuf[t * N + i] = sample(obsBuf[t], musBuf[i], sigmasBuf[i]);
        }
    }

    return p;
}


}

namespace discrete {

template<typename dtype, typename dtype_obs>
void updatePOut(const np_array<dtype_obs> &obs, const np_array<dtype> &weights, np_array<dtype> &pout) {
    auto T = static_cast<std::size_t>(obs.size());
    auto N = static_cast<std::size_t>(pout.shape(0));
    auto M = static_cast<std::size_t>(pout.shape(1));

    auto obsBuf = obs.data();
    auto weightsBuf = weights.data();
    auto poutBuf = pout.mutable_data();

    for(std::size_t t = 0; t < T; ++t) {
        auto o = obsBuf[t];
        for(std::size_t i = 0; i < N; ++i) {
            poutBuf[i * M + o] += weightsBuf[t*N + i];
        }
    }
}

template<typename dtype, typename STATE>
struct DiscreteOutputModel : public OutputModel<DiscreteOutputModel, dtype, STATE> {
    using Super = OutputModel<DiscreteOutputModel, dtype, STATE>;
    DiscreteOutputModel() : OutputModel<DiscreteOutputModel, dtype, STATE>(0, 0, true) {};

    explicit DiscreteOutputModel(np_array<dtype> outputProbabilityMatrix, py::object prior, bool outliers)
            : OutputModel<DiscreteOutputModel, dtype, STATE>(), outputProbabilityMatrix(outputProbabilityMatrix) {
        if(outputProbabilityMatrix.ndim() != 2) {
            throw std::invalid_argument("Discrete output model requires two-dimensional output probability matrix!");
        }
        auto nHidden = outputProbabilityMatrix.shape(0);
        auto nObs = outputProbabilityMatrix.shape(1);
        Super::setNHiddenStates(nHidden);
        Super::setNObservableStates(nObs);
        Super::setIgnoreOutliers(outliers);

        if(prior.is_none()) {
            this->_prior = np_array<dtype>({nHidden, nObs});
        } else {
            auto npPrior = py::cast<np_array<dtype>>(prior);
            if(npPrior.ndim() != 2) {
                throw std::invalid_argument("Discrete output model requires prior to be two-dimensional!");
            }
            if(npPrior.shape(0) != nHidden || npPrior.shape(1) != nObs) {
                std::stringstream ss;
                ss << "Requires prior to have shape ";
                ss << "(" << nHidden << ", " << nObs << ") ";
                ss << "but got ";
                ss << "(" << npPrior.shape(0) << ", " << npPrior.shape(1) << ")";
                throw std::invalid_argument(ss.str());
            }
            this->_prior = npPrior;
        }
        for(ssize_t row = 0; row < nObs; ++row) {
            dtype sum = 0;
            for(ssize_t col = 0; col < nHidden; ++col) {
                sum += outputProbabilityMatrix.at(col, row);
            }
            if(std::abs(sum - 1) > 1e-3) {
                throw std::invalid_argument("Output probability matrix is not row-stochastic.");
            }
        }
    }

    const np_array<dtype> &outputProbabilities() const {
        return outputProbabilityMatrix;
    }

    const np_array<dtype> &prior() const {
        return _prior;
    }

    void sample(const std::vector<np_array<STATE>> &observationsPerState) {
        auto nObs = Super::nObservableStates();
        ssize_t currentState {0};

        std::default_random_engine generator (clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
        dirichlet_distribution<dtype> dirichlet;

        for(const np_array<STATE> &observations : observationsPerState) {
            std::vector<dtype> hist (nObs, 0);
            for(ssize_t i = 0; i < observations.size(); ++i) {
                ++hist.at(observations.at(i));
            }
            auto priorBegin = _prior.data(currentState);
            // add prior onto histogram
            std::transform(hist.begin(), hist.end(), priorBegin, hist.begin(), std::plus<>());
            dirichlet.params(hist.begin(), hist.end());
            auto probs = dirichlet(generator);

            for(std::size_t i = 0; i < probs.size(); ++i) {
                if(probs[i] != 0) {
                    outputProbabilityMatrix.mutable_at(currentState, i) = probs[i];
                }
            }

            ++currentState;
        }
    }

    np_array<dtype> generateObservationTrajectoryImpl(const np_array<STATE> &hiddenStateTrajectory) const override {
        if(hiddenStateTrajectory.ndim() != 1) {
            throw std::invalid_argument("generate observation trajectory needs 1-dimensional hidden state trajectory");
        }
        auto nTimesteps = hiddenStateTrajectory.shape(0);
        auto nObs = Super::nObservableStates();

        np_array<dtype> output ({static_cast<std::size_t>(nTimesteps)});
        auto outputPtr = output.mutable_data();

        #pragma omp parallel
        {
            std::default_random_engine generator (clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
            std::discrete_distribution<> ddist;
            #pragma omp for
            for(decltype(nTimesteps) t = 0; t < nTimesteps; ++t) {
                auto state = hiddenStateTrajectory.at(t);
                auto begin = outputProbabilityMatrix.data(state, 0);
                auto end = begin + nObs;
                ddist.param(decltype(ddist)::param_type(begin, end));
                auto obs = ddist(generator);
                *(outputPtr + t) = obs;
            }
        }

        return output;
    }

    np_array<dtype> outputProbabilityTrajectoryImpl(const np_array<STATE> &observations) const override {
        if(observations.ndim() != 1) {
            throw std::invalid_argument("observations trajectory needs to be one-dimensional.");
        }
        auto nHidden = Super::nHiddenStates();
        np_array<dtype> output (std::vector<std::size_t>{static_cast<std::size_t>(observations.shape(0)),
                                                         nHidden});
        auto outputPtr = output.mutable_data(0);

        #pragma omp parallel for
        for(ssize_t t = 0; t < observations.shape(0); ++t) {
            auto obsState = observations.at(t);
            for(std::size_t i = 0; i < nHidden; ++i) {
                *(outputPtr + t * nHidden + i) = outputProbabilityMatrix.at(i, obsState);
            }
        }
        if(Super::ignoreOutliers()) {
            Super::handleOutliers(output);
        }
        return output;
    }

    DiscreteOutputModel submodelImpl(const np_array<STATE> &states) const override {
        if(states.ndim() != 1) {
            throw std::invalid_argument("Discrete output model submodel requires one-dimensional states array.");
        }
        np_array<dtype> restrictedOutputProbabilityMatrix({states.shape(0), outputProbabilityMatrix.shape(1)});
        np_array<dtype> restrictedPrior({states.shape(0), outputProbabilityMatrix.shape(1)});
        auto pptr = outputProbabilityMatrix.data();
        auto priorptr = _prior.data();
        for(ssize_t i = 0; i < states.shape(0); ++i) {
            auto state = states.at(i);
            std::copy(pptr + state * Super::nObservableStates(), pptr + (state+1) * Super::nObservableStates(),
                      restrictedOutputProbabilityMatrix.mutable_data(i, 0));
            std::copy(priorptr + state * Super::nObservableStates(), priorptr + (state+1) * Super::nObservableStates(),
                      restrictedPrior.mutable_data(i, 0));
        }
        return DiscreteOutputModel{restrictedOutputProbabilityMatrix, restrictedPrior, Super::_ignoreOutliers};
    }

private:
    np_array<dtype> outputProbabilityMatrix;
    np_array<dtype> _prior;
};


}

}
}
