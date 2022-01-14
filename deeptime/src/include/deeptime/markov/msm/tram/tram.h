//
// Created by Maaike on 15/11/2021.
//

#pragma once

#include <cstdio>
#include <cassert>
#include <utility>
#include <deeptime/numeric/kahan_summation.h>
#include "tram_input.h"

namespace deeptime::markov::tram {

namespace detail {

template<typename dtype>
constexpr static dtype prior() { return 0.0; }

template<typename dtype>
constexpr static dtype logPrior() { return -std::numeric_limits<dtype>::infinity(); }


template<typename dtype>
np_array_nfc<dtype> generateFilledArray(const std::vector<py::ssize_t> &dims, dtype fillValue) {
    np_array_nfc<dtype> array(dims);
    std::fill(array.mutable_data(), array.mutable_data() + array.size(), fillValue);
    return array;
}

template<py::ssize_t Dims, typename Array>
auto mutableBuf(Array &&array) {
    return array.template mutable_unchecked<Dims>();
}
}

template<typename dtype, py::ssize_t Dims>
class ExchangeableArray {
    // This is used as a helper class in TRAM for the case where a value updated, and for each update, the current and
    // previous values are stored. new values are generally computed in the first buffer, and to update the values,
    // exchange() is called, so that the first buffer becomes the second, and the second buffer the first, which can
    // subsequently be overwritten with the updated values. See e.g. use of thermStateEnergies_ in TRAM.
    using MutableBufferType = decltype(detail::mutableBuf<Dims>(std::declval<np_array<dtype>>()));
public:
    template<typename Shape = std::vector<py::ssize_t>>
    ExchangeableArray(Shape shape, dtype fillValue) : arrays(
            std::make_tuple(np_array<dtype>(shape), np_array<dtype>(shape))) {
        std::fill(std::get<0>(arrays).mutable_data(), std::get<0>(arrays).mutable_data() + std::get<0>(arrays).size(),
                  fillValue);
        std::fill(std::get<1>(arrays).mutable_data(), std::get<1>(arrays).mutable_data() + std::get<1>(arrays).size(),
                  fillValue);
        buffers = std::make_tuple(
                std::make_unique<MutableBufferType>(std::get<0>(arrays).template mutable_unchecked<Dims>()),
                std::make_unique<MutableBufferType>(std::get<1>(arrays).template mutable_unchecked<Dims>())
        );
    }

    ExchangeableArray(const ExchangeableArray &) = delete;

    ExchangeableArray &operator=(const ExchangeableArray &) = delete;

    void exchange() {
        current = !current;
    }

    auto *first() {
        return current ? &std::get<0>(arrays) : &std::get<1>(arrays);
    }

    const auto *first() const {
        return current ? &std::get<0>(arrays) : &std::get<1>(arrays);
    }

    auto *second() {
        return current ? &std::get<1>(arrays) : &std::get<0>(arrays);
    }

    const auto *second() const {
        return current ? &std::get<1>(arrays) : &std::get<0>(arrays);
    }

    auto &firstBuf() {
        return current ? *std::get<0>(buffers) : *std::get<1>(buffers);
    }

    const auto &firstBuf() const {
        return current ? *std::get<0>(buffers) : *std::get<1>(buffers);
    }

    auto &secondBuf() {
        return current ? *std::get<1>(buffers) : *std::get<0>(buffers);
    }

    const auto &secondBuf() const {
        return current ? *std::get<1>(buffers) : *std::get<0>(buffers);
    }

private:
    char current = 0;
    std::tuple<np_array<dtype>, np_array<dtype>> arrays;
    std::tuple<std::unique_ptr<MutableBufferType>, std::unique_ptr<MutableBufferType>> buffers;
};


// natural logarithm of the statistical weight per sample, log \mu^k(x).
// If thermState =-1, this is the unbiased statistical sample weight, log \mu(x).
template<typename dtype>
static const auto computeSampleWeightsLog(StateIndex thermStateIndex,
                                    const DTraj & dtraj,
                                    const BiasMatrix<dtype> & biasMatrix,
                                    const np_array<dtype> & thermStateEnergies,
                                    const np_array_nfc<dtype> & modifiedStateCountsLog) {
    std::size_t nThermStates = static_cast<StateIndex>(thermStateEnergies.size());
    std::size_t nMarkovStates = static_cast<StateIndex>(modifiedStateCountsLog.shape(1));

    std::vector<dtype> sampleWeights(dtraj.size());

    std::vector<dtype> scratch(nThermStates);

    ArrayBuffer<DTraj, 1> dtrajBuf{dtraj};
    ArrayBuffer<BiasMatrix<dtype>, 2> biasMatrixBuf{biasMatrix};
    ArrayBuffer<np_array_nfc<dtype>, 2> modifiedStateCountsLogBuf{modifiedStateCountsLog};
    ArrayBuffer<np_array_nfc<dtype>, 1> thermStateEnergiesBuf{thermStateEnergies};

    for (auto x = 0; x < dtraj.size(); ++x) {
        auto i = dtrajBuf(x);
        if (i < 0) {
            // The sample weight has no meaning, as it does not exist in the connected set. The value is set to -inf
            // so that the weight is zero and the sample does not contribute to the output when computing observables.
            sampleWeights[x] = -std::numeric_limits<dtype>::infinity();
            continue;
        }
        int o = 0;
        for (StateIndex l = 0; l < nThermStates; ++l) {
            if (modifiedStateCountsLogBuf(l, i) > -std::numeric_limits<dtype>::infinity()) {
                scratch[o++] = modifiedStateCountsLogBuf(l, i) - biasMatrixBuf(x, l);
            }
        }
        auto log_divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratch.begin(), o);
        if (thermStateIndex == -1) {// get unbiased sample weight
            sampleWeights[x] = -log_divisor;
        } else { // get biased sample weight for given thermState index
            sampleWeights[x] = -biasMatrixBuf(x, thermStateIndex) - log_divisor
                                        + thermStateEnergiesBuf(thermStateIndex);
        }
    }
    return sampleWeights;
}


// TRAM log-likelihood that comes from the terms containing discrete quantities.
// i.e. the likelihood of observing the observed transitions plus for each thermodynamic state,
// the free energy of that state times the state counts:
// \sum_{i,j,k}c_{ij}^{(k)}\log p_{ij}^{(k)} + \sum_{i,k}N_{i}^{(k)}f_{i}^{(k)}
template<typename EnergyBuffer, typename StateCountsBuffer, typename TransitionCountsBuffer, typename TransitionMatrixBuffer>
static const auto
computeDiscreteLikelihood(StateIndex nThermStates, StateIndex nMarkovStates, EnergyBuffer *biasedConfEnergies,
                          StateCountsBuffer *stateCounts, TransitionCountsBuffer *transitionCounts,
                          TransitionMatrixBuffer *transitionMatrices) {
    using dtype = typename EnergyBuffer::value_type;

    dtype LL = 0;

    #pragma omp parallel for default(none) firstprivate(nThermStates, nMarkovStates, transitionCounts, \
                                                            transitionMatrices, stateCounts, \
                                                            biasedConfEnergies) reduction(+:LL) collapse(2)
    for (StateIndex  k = 0; k < nThermStates; ++k) {
        for (StateIndex i = 0; i < nMarkovStates; ++i) {
            // discrete sample log-likelihood \sum_{k=1}^K \sum_{i=1}^m N_i^k * f_i^k
            if ((*stateCounts)(k, i) > 0) {
                LL += ((*stateCounts)(k, i) + tram::detail::prior<dtype>()) * (*biasedConfEnergies)(k, i);
            }
            // transition log-likelihood \sum_{k=1}^K \sum_{i,j=1}^m c_ij^k * log(p_ij^k)
            for (StateIndex j = 0; j < nMarkovStates; ++j) {
                auto CKij = (*transitionCounts)(k, i, j);
                if (CKij > 0) {
                    if (i == j) {
                        LL += (static_cast<dtype>(CKij) + tram::detail::prior<dtype>()) *
                              std::log((*transitionMatrices)(k, i, j));
                    } else {
                        LL += CKij * std::log((*transitionMatrices)(k, i, j));
                    }
                }
            }
        }
    }
    return LL;
}


// compute the log-likelihood of observing the input data under the current
// biasedConfEnergies_. This computes the parameter-dependent part of the likelihood, i.e. the factor -b^k(x) is
// omitted as it is constant.
template<typename dtype>
static const auto computeLogLikelihood (const DTraj &dtraj,
                                  const BiasMatrix<dtype> &biasMatrix,
                                  const np_array_nfc<dtype> &biasedConfEnergies,
                                  const np_array_nfc<dtype> &modifiedStateCountsLog,
                                  const np_array<dtype> &thermStateEnergies,
                                  const CountsMatrix &stateCounts,
                                  const CountsMatrix &transitionCounts,
                                  const np_array_nfc<dtype> &transitionMatrices) {
    auto nThermStates = static_cast<StateIndex>(biasedConfEnergies.shape(0));
    auto nMarkovStates = static_cast<StateIndex>(biasedConfEnergies.shape(1));

    // use threadsafe arraybuffers
    ArrayBuffer<np_array_nfc<dtype>, 2> biasedConfEnergiesBuf{biasedConfEnergies};
    ArrayBuffer<CountsMatrix, 3> transitionCountsBuf{transitionCounts};
    ArrayBuffer<CountsMatrix, 2> stateCountsBuf{stateCounts};
    ArrayBuffer<np_array_nfc<dtype>, 3> transitionMatricesBuf{transitionMatrices};

    // first get likelihood of all discrete quantities (transition likelihood and free energies times state counts)
    auto logLikelihood = computeDiscreteLikelihood(nThermStates, nMarkovStates, &biasedConfEnergiesBuf,
                                                   &stateCountsBuf, &transitionCountsBuf, &transitionMatricesBuf);

    // then compute log of all sample weights, and add to log likelihood.
    auto sampleWeights = computeSampleWeightsLog(-1, dtraj, biasMatrix, thermStateEnergies, modifiedStateCountsLog);
    logLikelihood += numeric::kahan::logsumexp_sort_kahan_inplace(sampleWeights.begin(), sampleWeights.end());

    return logLikelihood;
}

template<typename dtype>
struct TRAM {
public:

    TRAM(std::size_t nThermStates, std::size_t nMarkovStates)
            : nThermStates_(nThermStates),
              nMarkovStates_(nMarkovStates),
              biasedConfEnergies_(detail::generateFilledArray<dtype>({nThermStates_, nMarkovStates_}, 0.)),
              lagrangianMultLog_(ExchangeableArray<dtype, 2>({nThermStates_, nMarkovStates_}, 0.)),
              modifiedStateCountsLog_(detail::generateFilledArray<dtype>({nThermStates_, nMarkovStates_}, 0.)),
              thermStateEnergies_(ExchangeableArray<dtype, 1>(std::vector<StateIndex>{nThermStates_}, 0)),
              markovStateEnergies_(np_array_nfc<dtype>(std::vector<StateIndex>{nMarkovStates_})),
              transitionMatrices_(np_array_nfc<dtype>({nThermStates_, nMarkovStates_, nMarkovStates_})),
              statVectors_(ExchangeableArray<dtype, 2>(std::vector({nThermStates_, nMarkovStates_}), 0.)),
              scratch_(std::unique_ptr<dtype[]>(new dtype[std::max(nMarkovStates_, nThermStates_)])) {}


    TRAM(np_array_nfc<dtype> &biasedConfEnergies, np_array_nfc<dtype> &lagrangianMultLog,
         np_array_nfc<dtype> &modifiedStateCountsLog) : TRAM(biasedConfEnergies.shape(0), biasedConfEnergies.shape(1)) {
        std::copy(lagrangianMultLog.data(), lagrangianMultLog.data() + lagrangianMultLog.size(),
                  lagrangianMultLog_.first()->mutable_data());
        std::copy(biasedConfEnergies.data(), biasedConfEnergies.data() + biasedConfEnergies.size(),
                  biasedConfEnergies_.mutable_data());
        std::copy(modifiedStateCountsLog.data(), modifiedStateCountsLog.data() + modifiedStateCountsLog.size(),
                  modifiedStateCountsLog_.mutable_data());
    }

    const auto &biasedConfEnergies() const {
        return biasedConfEnergies_;
    }

    const auto &lagrangianMultLog() const {
        return *lagrangianMultLog_.first();
    }

    const auto &modifiedStateCountsLog() const {
        return modifiedStateCountsLog_;
    }

    const auto &thermStateEnergies() const {
        return *thermStateEnergies_.first();
    }

    const auto &markovStateEnergies() const {
        return markovStateEnergies_;
    }

    const auto &transitionMatrices() const {
        return transitionMatrices_;
    }


    // estimation loop to estimate the free energies. The current iteration number and iteration error are
    // returned through the callback function if it is provided. log-likelihoods are returned if
    // trackLogLikelihoods = true. By default, this is false, since calculating the log-likelihood required
    // an estimation of the transition matrices, which slows down estimation.
    void estimate(std::shared_ptr<TRAMInput<dtype>> &tramInput, std::size_t maxIter, dtype maxErr,
                  std::size_t callbackInterval = 1, bool trackLogLikelihoods = false,
                  const py::object *callback = nullptr) {

        input_ = tramInput;

        // initialize the lagrange multipliers with a default value based on the transition counts, but only
        // if all are zero.
        if (std::all_of(lagrangianMultLog().data(), lagrangianMultLog().data() + lagrangianMultLog().size(),
                        [](dtype x) { return x == static_cast<dtype>(0.); })) {
            initLagrangianMult();
        }

        double iterationError{0};

        for (decltype(maxIter) iterationCount = 0; iterationCount < maxIter; ++iterationCount) {

            // Self-consistent update of the TRAM equations.
            updateLagrangianMult();
            updateStateCounts();
            updateBiasedConfEnergies();

            // Tracking of energy vectors for error calculation.
            updateThermStateEnergies();
            updateStatVectors(statVectors_);

            // compare new thermStateEnergies_ and statVectors with old to get the
            // iteration error (= how much the energies changed).
            iterationError = computeError(statVectors_);

            dtype logLikelihood{0};
            if (trackLogLikelihoods) {
                computeTransitionMatrices();

                logLikelihood = computeLogLikelihood(input_->dtraj(), input_->biasMatrix(),
                                                     biasedConfEnergies_, modifiedStateCountsLog_,
                                                     *thermStateEnergies_.first(), input_->stateCounts(),
                                                     input_->transitionCounts(), transitionMatrices_);
            }

            if (callback != nullptr && callbackInterval > 0 && iterationCount % callbackInterval == 0) {
                py::gil_scoped_acquire guard;
                (*callback)(iterationError, logLikelihood);
            }

            if (iterationError < maxErr) {
                // We have converged!
                break;
            } else {
                // We are not finished. But before the next iteration, we shift all energies by min(energies)
                // so that the minimum energy equals zero (we are only interested in energy differences!).
                shiftEnergiesToHaveZeroMinimum();
            }
        }
        // Done iterating. Compute all energies for the thermodynamic states and markov states.
        updateMarkovStateEnergies();
        updateThermStateEnergies();
        normalize();

        // And compute final transition matrices
        computeTransitionMatrices();
    }

private:
    std::shared_ptr<TRAMInput<dtype>> input_;

    StateIndex nThermStates_;
    StateIndex nMarkovStates_;

    np_array_nfc<dtype> biasedConfEnergies_;
    ExchangeableArray<dtype, 2> lagrangianMultLog_;
    np_array_nfc<dtype> modifiedStateCountsLog_;

    ExchangeableArray<dtype, 1> thermStateEnergies_;
    np_array_nfc<dtype> markovStateEnergies_;
    np_array_nfc<dtype> transitionMatrices_;

    // Array to keep track of statVectors_(K, i) = exp(thermStateEnergies_(K) - biasedConfEnergies_(K, i))
    // difference between previous and current statvectors is used for calculating the iteration error to check whether
    // convergence is achieved.
    ExchangeableArray<dtype, 2> statVectors_;

    // scratch matrices used to facilitate calculation of logsumexp
    std::unique_ptr<dtype[]> scratch_;

    constexpr static dtype inf = std::numeric_limits<dtype>::infinity();

    constexpr static dtype prior() { return tram::detail::prior<dtype>(); }

    constexpr static dtype logPrior() { return tram::detail::logPrior<dtype>(); }


    void initLagrangianMult() {
        auto transitionCountsBuf = input_->transitionCountsBuf();
        auto lagrangianMultLogBuf = lagrangianMultLog_.firstBuf();

        auto nThermStates = nThermStates_;
        auto nMarkovStates = nMarkovStates_;

        #pragma omp parallel for default(none) firstprivate(nThermStates, nMarkovStates, \
                                                            transitionCountsBuf, lagrangianMultLogBuf)
        for (StateIndex k = 0; k < nThermStates; ++k) {
            for (StateIndex i = 0; i < nMarkovStates; ++i) {
                dtype sum = 0.0;
                for (StateIndex j = 0; j < nMarkovStates; ++j) {
                    sum += (transitionCountsBuf(k, i, j) + transitionCountsBuf(k, j, i));
                }
                lagrangianMultLogBuf(k, i) = std::log(sum / 2);
            }
        }
    }

    void updateLagrangianMult() {
        lagrangianMultLog_.exchange();
        auto oldLagrangianMultLogBuf = lagrangianMultLog_.secondBuf();
        auto newLagrangianMultLogBuf = lagrangianMultLog_.firstBuf();

        auto biasedConfEnergiesBuf = biasedConfEnergies_.template unchecked<2>();

        auto transitionCountsBuf = input_->transitionCountsBuf();
        auto stateCountsBuf = input_->stateCountsBuf();

        auto nThermStates = nThermStates_, nMarkovStates = nMarkovStates_;

        #pragma omp parallel for default(none) firstprivate(nThermStates, nMarkovStates, oldLagrangianMultLogBuf, \
                                                            newLagrangianMultLogBuf, biasedConfEnergiesBuf, \
                                                            transitionCountsBuf, stateCountsBuf)
        for (StateIndex k = 0; k < nThermStates; ++k) {
            std::vector<dtype> scratch(nMarkovStates);

            for (StateIndex i = 0; i < nMarkovStates; ++i) {
                if (0 == stateCountsBuf(k, i)) {
                    newLagrangianMultLogBuf(k, i) = -inf;
                    continue;
                }
                std::size_t o = 0;

                for (StateIndex j = 0; j < nMarkovStates; ++j) {
                    auto CKij = transitionCountsBuf(k, i, j);
                    // special case: most variables cancel out, here
                    if (i == j) {
                        scratch[o++] = (0 == CKij) ?
                                       logPrior() : std::log(prior() + (dtype) CKij);
                    } else {
                        auto CK = CKij + transitionCountsBuf(k, j, i);
                        if (0 != CK) {
                            auto divisor = numeric::kahan::logsumexp_pair<dtype>(
                                    oldLagrangianMultLogBuf(k, j) - biasedConfEnergiesBuf(k, i)
                                    - oldLagrangianMultLogBuf(k, i) + biasedConfEnergiesBuf(k, j), 0.0);
                            scratch[o++] = std::log((dtype) CK) - divisor;
                        }
                    }
                }
                newLagrangianMultLogBuf(k, i) = numeric::kahan::logsumexp_sort_kahan_inplace(scratch.begin(), o);
            }
        }
    }

    // update conformation energies based on one observed trajectory
    void updateBiasedConfEnergies() {
        std::fill(biasedConfEnergies_.mutable_data(), biasedConfEnergies_.mutable_data() +
                                                      biasedConfEnergies_.size(), inf);

        auto biasMatrixBuf = input_->biasMatrixBuf();
        auto dtrajBuf = input_->dtrajBuf();

        auto biasedConfEnergiesBuf = biasedConfEnergies_.template mutable_unchecked<2>();
        auto modifiedStateCountsLogBuf = modifiedStateCountsLog_.template unchecked<2>();

        auto *scratch = scratch_.get();

        // assume that biasedConfEnergies_ have been set to INF by the caller in the first call
        for (std::int32_t x = 0; x < input_->nSamples(); ++x) {
            auto i = dtrajBuf(x);
            if (i >= 0) { // skip frames that have negative Markov state indices
                std::size_t o = 0;
                for (StateIndex k = 0; k < nThermStates_; ++k) {
                    if (modifiedStateCountsLogBuf(k, i) > -inf) {
                        scratch[o++] = modifiedStateCountsLogBuf(k, i) - biasMatrixBuf(x, k);
                    }
                }
                dtype divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratch, o);

                for (StateIndex k = 0; k < nThermStates_; ++k) {
                    biasedConfEnergiesBuf(k, i) = -numeric::kahan::logsumexp_pair(
                            -biasedConfEnergiesBuf(k, i), -(divisor + biasMatrixBuf(x, k)));
                }
            }
        }
    }


    void updateStateCounts() {
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template unchecked<2>();
        auto lagrangianMultLogBuf = lagrangianMultLog_.firstBuf();
        auto modifiedStateCountsLogBuf = modifiedStateCountsLog_.template mutable_unchecked<2>();

        auto stateCountsBuf = input_->stateCountsBuf();
        auto transitionCountsBuf = input_->transitionCountsBuf();

        #pragma omp parallel for default(none) firstprivate(biasedConfEnergiesBuf, lagrangianMultLogBuf, \
                                                            modifiedStateCountsLogBuf, stateCountsBuf, \
                                                            transitionCountsBuf) collapse(2)
        for (StateIndex k = 0; k < nThermStates_; ++k) {
            for (StateIndex i = 0; i < nMarkovStates_; ++i) {

                std::vector<dtype> scratch;
                scratch.reserve(nMarkovStates_);

                if (0 == stateCountsBuf(k, i)) {
                    modifiedStateCountsLogBuf(k, i) = -inf;
                } else {
                    auto Ci = 0;
                    for (StateIndex j = 0; j < nMarkovStates_; ++j) {
                        auto CKij = transitionCountsBuf(k, i, j);
                        auto CKji = transitionCountsBuf(k, j, i);
                        Ci += CKji;
                        // special case: most variables cancel out, here
                        if (i == j) {
                            auto CKijLog = (0 == CKij) ? logPrior() : std::log(prior() + (dtype) CKij);
                            scratch.push_back(CKijLog + biasedConfEnergiesBuf(k, i));
                        } else {
                            auto CK = CKij + CKji;

                            if (CK > 0) {
                                auto divisor = numeric::kahan::logsumexp_pair(
                                        lagrangianMultLogBuf(k, j) - biasedConfEnergiesBuf(k, i),
                                        lagrangianMultLogBuf(k, i) - biasedConfEnergiesBuf(k, j));
                                scratch.push_back(std::log((dtype) CK) + lagrangianMultLogBuf(k, j) - divisor);
                            }
                        }
                    }
                    auto NC = stateCountsBuf(k, i) - Ci;
                    auto extraStateCounts = (0 < NC) ? std::log((dtype) NC) + biasedConfEnergiesBuf(k, i) : -inf;
                    modifiedStateCountsLogBuf(k, i) = numeric::kahan::logsumexp_pair(
                            numeric::kahan::logsumexp_sort_kahan_inplace(scratch.begin(), scratch.end()),
                            extraStateCounts);
                }
            }
        }
    }

    // Get the error in the energies between this iteration and the previous one.
    dtype computeError(const ExchangeableArray<dtype, 2> &statVectors) const {
        auto thermEnergiesBuf = thermStateEnergies_.firstBuf();
        auto oldThermEnergiesBuf = thermStateEnergies_.secondBuf();
        auto statVectorsBuf = statVectors.firstBuf();
        auto oldStatVectorsBuf = statVectors.secondBuf();

        dtype maxError = 0;
        auto nThermStates = nThermStates_;
        auto nMarkovStates = nMarkovStates_;

        #pragma omp parallel for default(none) shared(maxError) firstprivate(nThermStates, nMarkovStates, \
                                    thermEnergiesBuf, oldThermEnergiesBuf, statVectorsBuf, oldStatVectorsBuf)
        for (StateIndex k = 0; k < nThermStates; ++k) {
            auto energyDelta = std::abs(thermEnergiesBuf(k) - oldThermEnergiesBuf(k));
            maxError = std::max(maxError, energyDelta);

            for (StateIndex i = 0; i < nMarkovStates; ++i) {
                energyDelta = std::abs(statVectorsBuf(k, i) - oldStatVectorsBuf(k, i));
                maxError = std::max(maxError, energyDelta);
            }
        }
        return maxError;
    }

    void updateStatVectors(ExchangeableArray<dtype, 2> &statVectors) {
        // move current values to old
        statVectors.exchange();

        // compute new values
        auto statVectorsBuf = statVectors.firstBuf();
        auto thermStateEnergiesBuf = thermStateEnergies_.firstBuf();
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template unchecked<2>();

        #pragma omp parallel for default(none) firstprivate(statVectorsBuf, \
                                                            thermStateEnergiesBuf, biasedConfEnergiesBuf)
        for (StateIndex k = 0; k < nThermStates_; ++k) {
            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                statVectorsBuf(k, i) = std::exp(thermStateEnergiesBuf(k) - biasedConfEnergiesBuf(k, i));
            }
        }
    }

    void updateMarkovStateEnergies() {
        // first reset all confirmation energies to infinity
        std::fill(markovStateEnergies_.mutable_data(), markovStateEnergies_.mutable_data() + nMarkovStates_, inf);

        auto modifiedStateCountsLogBuf = modifiedStateCountsLog_.template unchecked<2>();
        auto markovStateEnergiesBuf = markovStateEnergies_.template mutable_unchecked<1>();

        auto dtrajBuf = input_->dtrajBuf();
        auto biasMatrixBuf = input_->biasMatrixBuf();

        // assume that markovStateEnergies_ were set to INF by the caller on the first call
        for (auto x = 0; x < input_->nSamples(); ++x) {
            std::int32_t i = dtrajBuf(x);
            if (i >= 0) { // skip negative state indices
                std::size_t o = 0;
                for (StateIndex k = 0; k < nThermStates_; ++k) {
                    if (modifiedStateCountsLogBuf(k, i) > -inf) {
                        scratch_[o++] = modifiedStateCountsLogBuf(k, i) - biasMatrixBuf(x, k);
                    }
                }
                dtype divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratch_.get(), o);
                markovStateEnergiesBuf(i) = -numeric::kahan::logsumexp_pair(-markovStateEnergiesBuf(i), -divisor);
            }
        }
    }


    void updateThermStateEnergies() {
        // move current values to old
        thermStateEnergies_.exchange();

        // compute new
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template unchecked<2>();
        auto thermStateEnergiesBuf = thermStateEnergies_.firstBuf();
        auto scratch = scratch_.get();

        for (StateIndex k = 0; k < nThermStates_; ++k) {
            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                scratch[i] = -biasedConfEnergiesBuf(k, i);
            }
            thermStateEnergiesBuf(k) = -numeric::kahan::logsumexp_sort_kahan_inplace(scratch, nMarkovStates_);
        }
    }

    // Shift all energies by min(biasedConfEnergies_) so the energies don't drift to
    // very large values.
    void shiftEnergiesToHaveZeroMinimum() {
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template mutable_unchecked<2>();
        auto thermStateEnergiesBuf = thermStateEnergies_.firstBuf();

        auto ptr = biasedConfEnergies_.data();
        auto shift = *std::min_element(ptr, ptr + biasedConfEnergies_.size());

        #pragma omp parallel for default(none) firstprivate(biasedConfEnergiesBuf, thermStateEnergiesBuf, shift)
        for (StateIndex k = 0; k < nThermStates_; ++k) {
            thermStateEnergiesBuf(k) -= shift;

            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                biasedConfEnergiesBuf(k, i) -= shift;
            }
        }
    }

    void normalize() {
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template mutable_unchecked<2>();
        auto markovStateEnergiesBuf = markovStateEnergies_.template mutable_unchecked<1>();
        auto thermStateEnergiesBuf = thermStateEnergies_.firstBuf();

        for (StateIndex i = 0; i < nMarkovStates_; ++i) {
            scratch_[i] = -markovStateEnergiesBuf(i);
        }
        auto f0 = -numeric::kahan::logsumexp_sort_kahan_inplace(scratch_.get(), nMarkovStates_);

        for (StateIndex i = 0; i < nMarkovStates_; ++i) {
            markovStateEnergiesBuf(i) -= f0;
            for (StateIndex k = 0; k < nThermStates_; ++k) {
                biasedConfEnergiesBuf(k, i) -= f0;
            }
        }
        for (StateIndex k = 0; k < nThermStates_; ++k) {
            thermStateEnergiesBuf(k) -= f0;
        }

        // update the state counts because they also include biased conf energies.
        // If this is not done after normalizing, the log likelihood computations will
        // not produce the correct output, due to incorrect values for mu(x).
        updateStateCounts();
    }

    void computeTransitionMatrices() {
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template unchecked<2>();
        auto lagrangianMultLogBuf = lagrangianMultLog_.firstBuf();

        auto transitionCountsBuf = input_->transitionCountsBuf();
        auto transitionMatricesBuf = transitionMatrices_.template mutable_unchecked<3>();

        auto nThermStates = nThermStates_;
        auto nMarkovStates = nMarkovStates_;

        #pragma omp parallel for default(none) firstprivate(nThermStates, nMarkovStates, biasedConfEnergiesBuf, \
                                                            lagrangianMultLogBuf, transitionCountsBuf, \
                                                            transitionMatricesBuf)
        for (StateIndex k = 0; k < nThermStates; ++k) {
            std::vector<dtype> scratch(nMarkovStates, 0);
            for (StateIndex i = 0; i < nMarkovStates; ++i) {
                for (StateIndex j = 0; j < nMarkovStates; ++j) {
                    transitionMatricesBuf(k, i, j) = 0.0;
                    auto C = transitionCountsBuf(k, i, j) + transitionCountsBuf(k, j, i);

                    if (C > 0) { // skip if there were no transitions
                        if (i == j) {
                            // special case: diagonal element
                            transitionMatricesBuf(k, i, j) = 0.5 * C * exp(-lagrangianMultLogBuf(k, i));
                        } else {
                            // regular case
                            auto divisor = numeric::kahan::logsumexp_pair(
                                    lagrangianMultLogBuf(k, j) - biasedConfEnergiesBuf(k, i),
                                    lagrangianMultLogBuf(k, i) - biasedConfEnergiesBuf(k, j));
                            transitionMatricesBuf(k, i, j) = C * exp(-(biasedConfEnergiesBuf(k, j) + divisor));
                        }
                        scratch[i] += transitionMatricesBuf(k, i, j);
                    }
                }
            }
            // normalize transition matrix
            auto maxSumIt = std::max_element(std::begin(scratch), std::end(scratch));

            dtype maxSum;
            if (maxSumIt == std::end(scratch) || *maxSumIt == 0) {
                maxSum = 1.0; // completely empty T matrix -> generate Id matrix
            } else {
                maxSum = *maxSumIt;
            }

            for (StateIndex i = 0; i < nMarkovStates; ++i) {
                for (StateIndex j = 0; j < nMarkovStates; ++j) {
                    if (i == j) {
                        transitionMatricesBuf(k, i, i) =
                                (transitionMatricesBuf(k, i, i) + maxSum - scratch[i]) / maxSum;
                        if (0 == transitionMatricesBuf(k, i, i) && 0 < transitionCountsBuf(k, i, i)) {
                            std::stringstream ss;
                            ss << "# Warning: zero diagonal element T[" << i << "," << i << "] with non-zero counts.";
                            #pragma omp critical
                            {
                                py::gil_scoped_acquire gil;
                                py::print(ss.str());
                            }
                        }
                    } else {
                        transitionMatricesBuf(k, i, j) = transitionMatricesBuf(k, i, j) / maxSum;
                    }
                }
            }
        }
    }
};
}