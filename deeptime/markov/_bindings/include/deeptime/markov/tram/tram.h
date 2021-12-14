//
// Created by Maaike on 15/11/2021.
//

#pragma once

#include <cstdio>
#include <cassert>
#include <utility>
#include <pybind11/stl.h>
#include <deeptime/common.h>
#include "kahan_summation.h"


namespace deeptime::tram {

namespace detail {
constexpr void throwIfInvalid(bool isValid, const std::string &message) {
    if (!isValid) {
        throw std::runtime_error(message);
    }
}

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


template<typename dtype>
class TRAMInput {
public:
    using DTraj = np_array<std::int32_t>;
    using DTrajs = std::vector<DTraj>;

    using BiasMatrix = np_array_nfc<dtype>;
    using BiasMatrices = std::vector<BiasMatrix>;

    TRAMInput(np_array_nfc<std::int32_t> &&stateCounts, np_array_nfc<std::int32_t> &&transitionCounts,
              DTrajs dtrajs, BiasMatrices biasMatrices)
            : stateCounts_(std::move(stateCounts)),
              transitionCounts_(std::move(transitionCounts)),
              dtrajs_(std::move(dtrajs)),
              _biasMatrices(std::move(biasMatrices)) {
        validateInput();
    }

    TRAMInput() = default;

    TRAMInput(const TRAMInput &) = delete;

    TRAMInput &operator=(const TRAMInput &) = delete;

    TRAMInput(TRAMInput &&) noexcept = default;

    TRAMInput &operator=(TRAMInput &&) noexcept = default;

    ~TRAMInput() = default;

    void validateInput() const {

        if (dtrajs_.size() != _biasMatrices.size()) {
            std::stringstream ss;
            ss << "Input invalid. Number of trajectories should be equal to the size of the first dimension "
                  "of the bias matrix.";
            ss << "\nNumber of trajectories: " << dtrajs_.size() << "\nNumber of bias matrices: "
               << _biasMatrices.size();
            throw std::runtime_error(ss.str());
        }
        detail::throwIfInvalid(stateCounts_.shape(0) == transitionCounts_.shape(0),
                               "stateCounts.shape(0) should equal transitionCounts.shape(0)");
        detail::throwIfInvalid(stateCounts_.shape(1) == transitionCounts_.shape(1),
                               "stateCounts.shape(1) should equal transitionCounts.shape(1)");
        detail::throwIfInvalid(transitionCounts_.shape(1) == transitionCounts_.shape(2),
                               "transitionCounts.shape(1) should equal transitionCounts.shape(2)");

        for (std::size_t i = 0; i < dtrajs_.size(); ++i) {
            const auto &dtraj = dtrajs_.at(i);
            const auto &biasMatrix = _biasMatrices.at(i);

            detail::throwIfInvalid(dtraj.ndim() == 1,
                                   "dtraj at index {i} has an incorrect number of dimension. ndims should be 1.");
            detail::throwIfInvalid(biasMatrix.ndim() == 2,
                                   "biasMatrix at index {i} has an incorrect number of dimension. ndims should be 2.");
            detail::throwIfInvalid(dtraj.shape(0) == biasMatrix.shape(0),
                                   "dtraj and biasMatrix at index {i} should be of equal length.");
            detail::throwIfInvalid(biasMatrix.shape(1) == transitionCounts_.shape(0),
                                   "biasMatrix{i}.shape[1] should be equal to transitionCounts.shape[0].");
        }
    }

    auto biasMatrix(std::size_t i) const {
        return _biasMatrices.at(i).template unchecked<2>();
    }

    auto dtraj(std::size_t i) const {
        return dtrajs_[i].template unchecked<1>();
    }

    auto transitionCounts() const {
        return transitionCounts_.template unchecked<3>();
    }

    auto stateCounts() const {
        return stateCounts_.template unchecked<2>();
    }

    auto sequenceLength(std::size_t i) const {
        return dtrajs_[i].size();
    }

    auto nTrajectories() const {
        return dtrajs_.size();
    };

private:
    np_array_nfc<std::int32_t> stateCounts_;
    np_array_nfc<std::int32_t> transitionCounts_;
    DTrajs dtrajs_;
    BiasMatrices _biasMatrices;
};


template<typename dtype>
struct TRAM {
public:

    using StateIndex = py::ssize_t;
    TRAM(std::shared_ptr<TRAMInput<dtype>> tramInput, std::size_t callbackInterval)
            : input_(tramInput),
              nThermStates_(tramInput->stateCounts().shape(0)),
              nMarkovStates_(tramInput->stateCounts().shape(1)),
              callbackInterval_(callbackInterval),
              biasedConfEnergies_(detail::generateFilledArray<dtype>({nThermStates_, nMarkovStates_}, 0.)),
              lagrangianMultLog_(ExchangeableArray<dtype, 2>({nThermStates_, nMarkovStates_}, 0.)),
              modifiedStateCountsLog_(detail::generateFilledArray<dtype>({nThermStates_, nMarkovStates_}, 0.)),
              thermStateEnergies_(ExchangeableArray<dtype, 1>(std::vector<StateIndex>{nThermStates_}, 0)),
              markovStateEnergies_(np_array_nfc<dtype>(std::vector<StateIndex>{nMarkovStates_})),
              transitionMatrices_(np_array_nfc<dtype>({nThermStates_, nMarkovStates_, nMarkovStates_})),
              scratchM_(std::unique_ptr<dtype[]>(new dtype[nMarkovStates_])),
              scratchT_(std::unique_ptr<dtype[]>(new dtype[nThermStates_])) {
        initLagrangianMult();
    }

    auto energiesPerThermodynamicState() const {
        return *thermStateEnergies_.first();
    }

    auto energiesPerMarkovState() const {
        return markovStateEnergies_;
    }

    auto biasedConfEnergies() const {
        return biasedConfEnergies_;
    }

    auto transitionMatrices() {
        return transitionMatrices_;
    }

    // compute the log-likelihood of observing the input data under the current
    // biasedConfEnergies_.
    dtype computeLogLikelihood() const {
        dtype logLikelihood = 0.;

        logLikelihood += computeTransitionLikelihood();

        auto nTraj = input_->nTrajectories();
        #pragma omp parallel for default(none) firstprivate(nTraj) reduction(+:logLikelihood)
        for (auto i = 0; i < nTraj; ++i) {
            logLikelihood += computeSampleLikelihood(i);
        }

        return logLikelihood;
    }


    // estimation loop to estimate the free energies. The current iteration number and iteration error are
    // returned through the callback function if it is provided. log-likelihoods are returned if
    // trackLogLikelihoods = true. By default, this is false, since calculating the log-likelihood required
    // an estimation of the transition matrices, which slows down estimation.
    void estimate(std::size_t maxIter, dtype maxErr, bool trackLogLikelihoods = false,
                  const py::object *callback = nullptr) {

        // Array to keep track of _statVectors(K, i) = exp(_thermStateEnergies(K) - _biasedConfEnergies(K, i))
        // difference between previous and current statvectors is used for calculting the iteration error.
        auto statVectors = ExchangeableArray<dtype, 2>(std::vector({nThermStates_, nMarkovStates_}), 0.);
        auto iterationError = 0.;

        for (decltype(maxIter) iterationCount = 0; iterationCount < maxIter; ++iterationCount) {

            // Self-consistent update of the TRAM equations.
            updateLagrangianMult();
            updateStateCounts();
            updateBiasedConfEnergies();

            // Tracking of energy vectors for error calculation.
            updateThermStateEnergies();
            updateStatVectors(statVectors);

            // compare new thermStateEnergies_ and statVectors with old to get the
            // iteration error (= how much the energies changed).
            iterationError = getError(statVectors);

            dtype logLikelihood {0};
            if (trackLogLikelihoods) {
                updateTransitionMatrices();
                logLikelihood = computeLogLikelihood();
            }

            if (callback != nullptr && iterationCount % callbackInterval_ == 0) {
                // TODO: callback doesn't work in release???
                py::gil_scoped_acquire guard;
                (*callback)(iterationCount, iterationError, logLikelihood);
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
        computeMarkovStateEnergies();
        updateThermStateEnergies();
        normalize();

        // And update final transition matrices
        updateTransitionMatrices();

        if (iterationError >= maxErr) {
            // We exceeded maxIter but we did not converge.
            std::cout << "TRAM did not converge. Last increment = " << iterationError << std::endl;
        }
    }

    std::vector<np_array_nfc<dtype>> getSampleWeights(std::int32_t thermState) {
        std::vector<np_array_nfc<dtype>> sampleWeights(nThermStates_);

        for (std::size_t i = 0; i < input_->nTrajectories(); ++i) {
            sampleWeights.push_back(getSampleWeightsForTrajectory(i, thermState));
        }
        return sampleWeights;
    }

private:
    std::shared_ptr<TRAMInput<dtype>> input_;

    StateIndex nThermStates_;
    StateIndex nMarkovStates_;

    std::size_t callbackInterval_;

    np_array_nfc<dtype> biasedConfEnergies_;
    ExchangeableArray<dtype, 2> lagrangianMultLog_;
    np_array_nfc<dtype> modifiedStateCountsLog_;

    ExchangeableArray<dtype, 1> thermStateEnergies_;
    np_array_nfc<dtype> markovStateEnergies_;
    np_array_nfc<dtype> transitionMatrices_;

    // scratch matrices used to facilitate calculation of logsumexp
    std::unique_ptr<dtype[]> scratchM_;
    std::unique_ptr<dtype[]> scratchT_;


    dtype inf = std::numeric_limits<dtype>::infinity();


    constexpr static dtype prior() { return 0.0; }

    constexpr static dtype logPrior() { return 1.0; }

    void initLagrangianMult() {
        auto transitionCountsBuf = input_->transitionCounts();
        auto lagrangianMultLogBuf = lagrangianMultLog_.firstBuf();

        #pragma omp parallel for default(none) firstprivate(nThermStates_, nMarkovStates_, transitionCountsBuf, lagrangianMultLogBuf)
        for (StateIndex k = 0; k < nThermStates_; ++k) {
            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                dtype sum = 0.0;
                for (StateIndex j = 0; j < nMarkovStates_; ++j) {
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

        auto transitionCountsBuf = input_->transitionCounts();
        auto stateCountsBuf = input_->stateCounts();

        for (StateIndex k = 0; k < nThermStates_; ++k) {
            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                if (0 == stateCountsBuf(k, i)) {
                    newLagrangianMultLogBuf(k, i) = -inf;
                    continue;
                }
                std::size_t o = 0;

                for (StateIndex j = 0; j < nMarkovStates_; ++j) {
                    auto CKij = transitionCountsBuf(k, i, j);
                    // special case: most variables cancel out, here
                    if (i == j) {
                        scratchM_[o++] = (0 == CKij) ?
                                         logPrior() : std::log(prior() + (dtype) CKij);
                    } else {
                        auto CK = CKij + transitionCountsBuf(k, j, i);
                        // todo why?
                        if (0 != CK) {
                            auto divisor = numeric::kahan::logsumexp_pair<dtype>(
                                    oldLagrangianMultLogBuf(k, j) - biasedConfEnergiesBuf(k, i)
                                    - oldLagrangianMultLogBuf(k, i) + biasedConfEnergiesBuf(k, j), 0.0);
                            scratchM_[o++] = std::log((dtype) CK) - divisor;
                        }
                    }
                }
                newLagrangianMultLogBuf(k, i) = numeric::kahan::logsumexp_sort_kahan_inplace(scratchM_.get(), o);
            }
        }
    }

    // update all biased confirmation energies by looping over all trajectories.
    void updateBiasedConfEnergies() {
        std::fill(biasedConfEnergies_.mutable_data(), biasedConfEnergies_.mutable_data() +
                                                      nMarkovStates_ * nThermStates_, inf);

        for (std::size_t i = 0; i < input_->nTrajectories(); ++i) {
            updateBiasedConfEnergies(i, input_->sequenceLength(i));
        }
    }

    // update conformation energies based on one observed trajectory
    void updateBiasedConfEnergies(std::size_t thermState, std::size_t trajLength) {
        auto biasMatrixBuf = input_->biasMatrix(thermState);
        auto dtrajBuf = input_->dtraj(thermState);

        auto biasedConfEnergiesBuf = biasedConfEnergies_.template mutable_unchecked<2>();
        auto modifiedStateCountsLogBuf = modifiedStateCountsLog_.template unchecked<2>();

        dtype divisor{};

        auto *scratch = scratchT_.get();

        // assume that biasedConfEnergies_ have been set to INF by the caller in the first call
        for (std::size_t x = 0; x < trajLength; ++x) {
            auto i = dtrajBuf(x);
            if (i >= 0) { // skip frames that have negative Markov state indices
                std::size_t o = 0;
                for (StateIndex k = 0; k < nThermStates_; ++k) {
                    if (modifiedStateCountsLogBuf(k, i) > -inf) {
                        scratch[o++] = modifiedStateCountsLogBuf(k, i) - biasMatrixBuf(x, k);
                    }
                }
                divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratch, o);

                for (StateIndex k = 0; k < nThermStates_; ++k) {
                    biasedConfEnergiesBuf(k, i) = -numeric::kahan::logsumexp_pair(
                            -biasedConfEnergiesBuf(k, i), -(divisor + biasMatrixBuf(x, k)));
                }
            }
        }
    }


//    template<typename dtype, bool trammbar = false>
    void updateStateCounts() {
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template unchecked<2>();
        auto lagrangianMultLogBuf = lagrangianMultLog_.firstBuf();
        auto modifiedStateCountsLogBuf = modifiedStateCountsLog_.template mutable_unchecked<2>();

        auto stateCountsBuf = input_->stateCounts();
        auto transitionCountsBuf = input_->transitionCounts();

        for (StateIndex k = 0; k < nThermStates_; ++k) {
            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                if (0 == stateCountsBuf(k, i)) {
                    modifiedStateCountsLogBuf(k, i) = -inf;
                } else {
                    auto Ci = 0;
                    auto o = 0;
                    for (StateIndex j = 0; j < nMarkovStates_; ++j) {
                        auto CKij = transitionCountsBuf(k, i, j);
                        auto CKji = transitionCountsBuf(k, j, i);
                        Ci += CKji;
                        // special case: most variables cancel out, here
                        if (i == j) {
                            scratchM_[o] = (0 == CKij) ? logPrior() : std::log(
                                    prior() + (dtype) CKij);
                            scratchM_[o++] += biasedConfEnergiesBuf(k, i);
                        } else {
                            auto CK = CKij + CKji;

                            if (CK > 0) {
                                auto divisor = numeric::kahan::logsumexp_pair(
                                        lagrangianMultLogBuf(k, j) - biasedConfEnergiesBuf(k, i),
                                        lagrangianMultLogBuf(k, i) - biasedConfEnergiesBuf(k, j));
                                scratchM_[o++] = std::log((dtype) CK) + lagrangianMultLogBuf(k, j) - divisor;
                            }
                        }
                    }
                    auto NC = stateCountsBuf(k, i) - Ci;
                    auto extraStateCounts = (0 < NC) ? std::log((dtype) NC) + biasedConfEnergiesBuf(k, i) : -inf;
                    modifiedStateCountsLogBuf(k, i) = numeric::kahan::logsumexp_pair(
                            numeric::kahan::logsumexp_sort_kahan_inplace(scratchM_.get(), o), extraStateCounts);
                }
            }
        }
    }

    // Get the error in the energies between this iteration and the previous one.
    dtype getError(ExchangeableArray<dtype, 2> &statVectors) {
        auto thermEnergiesBuf = thermStateEnergies_.firstBuf();
        auto oldThermEnergiesBuf = thermStateEnergies_.secondBuf();
        auto statVectorsBuf = statVectors.firstBuf();
        auto oldStatVectorsBuf = statVectors.secondBuf();

        dtype maxError = 0;

        for (StateIndex k = 0; k < nThermStates_; ++k) {
            auto energyDelta = std::abs(thermEnergiesBuf(k) - oldThermEnergiesBuf(k));
            if (energyDelta > maxError) maxError = energyDelta;

            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                energyDelta = std::abs(statVectorsBuf(k, i) - oldStatVectorsBuf(k, i));
                if (energyDelta > maxError) maxError = energyDelta;
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

        for (StateIndex k = 0; k < nThermStates_; ++k) {
            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                statVectorsBuf(k, i) = std::exp(thermStateEnergiesBuf(k) - biasedConfEnergiesBuf(k, i));
            }
        }
    }

    void computeMarkovStateEnergies() {
        // first reset all confirmation energies to infinity
        std::fill(markovStateEnergies_.mutable_data(), markovStateEnergies_.mutable_data() + nMarkovStates_, inf);

        for (std::size_t i = 0; i < input_->nTrajectories(); ++i) {
            auto dtraj = input_->dtraj(i);
            auto biasMatrix = input_->biasMatrix(i);

            computeMarkovStateEnergiesForSingleTrajectory(biasMatrix, dtraj);
        }
    }

    template<typename BiasMatrix, typename Dtraj>
    void computeMarkovStateEnergiesForSingleTrajectory(const BiasMatrix &biasMatrix, const Dtraj &dtraj) {
        auto modifiedStateCountsLogBuf = modifiedStateCountsLog_.template unchecked<2>();
        auto markovStateEnergiesBuf = markovStateEnergies_.template mutable_unchecked<1>();

        dtype divisor;
        // assume that markovStateEnergies_ were set to INF by the caller on the first call
        for (auto x = 0; x < dtraj.size(); ++x) {
            std::int32_t i = dtraj(x);
            if (i >= 0) { // skip negative state indices
                std::size_t o = 0;
                for (StateIndex k = 0; k < nThermStates_; ++k) {
                    if (modifiedStateCountsLogBuf(k, i) > -inf) {
                        scratchT_[o++] = modifiedStateCountsLogBuf(k, i) - biasMatrix(x, k);
                    }
                }
                divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratchT_.get(), o);
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
        auto scratch = scratchM_.get();

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

        dtype shift = 0;

        for (StateIndex k = 0; k < nThermStates_; ++k) {
            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                if (biasedConfEnergiesBuf(k, i) < shift) {
                    shift = biasedConfEnergiesBuf(k, i);
                }
            }
        }
        for (StateIndex K = 0; K < nThermStates_; ++K) {
            thermStateEnergiesBuf(K) -= shift;

            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                biasedConfEnergiesBuf(K, i) -= shift;
            }
        }
    }

    void normalize() {
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template mutable_unchecked<2>();
        auto markovStateEnergiesBuf = markovStateEnergies_.template mutable_unchecked<1>();
        auto thermStateEnergiesBuf = thermStateEnergies_.firstBuf();

        for (StateIndex i = 0; i < nMarkovStates_; ++i) {
            scratchM_[i] = -markovStateEnergiesBuf(i);
        }
        dtype f0 = -numeric::kahan::logsumexp_sort_kahan_inplace(scratchM_.get(), nMarkovStates_);

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

    // log likelihood of observing a sampled trajectory from the local equilibrium.
    // i.e. the sum over all sample likelihoods from one trajectory.
    // -\sum_{x}\log\sum_{l}R_{i(x)}^{(l)}e^{-b^{(l)}(x)+f_{i(x)}^{(l)}}
    dtype computeSampleLikelihood(std::size_t trajIndex) const {
        auto modifiedStateCountsLogBuf = modifiedStateCountsLog_.template unchecked<2>();
        auto biasMatrix = input_->biasMatrix(trajIndex);
        auto dtraj = input_->dtraj(trajIndex);

        std::vector<dtype> scratch;
        scratch.reserve(nThermStates_);

        dtype logLikelihood = 0;
        for (std::size_t x = 0; x < dtraj.size(); ++x) {
            auto o = 0;
            auto i = dtraj(x);

            if (i < 0) continue; // skip negative markov state indices

            for (StateIndex k = 0; k < nThermStates_; ++k) {
                if (modifiedStateCountsLogBuf(k, i) > 0) {
                    scratch.push_back(modifiedStateCountsLogBuf(k, i) - biasMatrix(x, k));
                }
            }
            logLikelihood -= numeric::kahan::logsumexp_sort_kahan_inplace(begin(scratch), end(scratch));
        }
        return logLikelihood;
    }


    // TRAM log-likelihood that comes from the terms containing discrete quantities.
    // i.e. the likelihood of observing the observed transitions.
    dtype computeTransitionLikelihood() const {
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template unchecked<2>();
        auto transitionCountsBuf = input_->transitionCounts();
        auto stateCountsBuf = input_->stateCounts();
        auto transitionMatricesBuf = transitionMatrices_.template unchecked<3>();

        // \sum_{i,j,k}c_{ij}^{(k)}\log p_{ij}^{(k)} + \sum_{i,k}N_{i}^{(k)}f_{i}^{(k)}
        dtype a = 0;

        #pragma omp parallel for default(none) firstprivate(nThermStates_, nMarkovStates_, transitionCountsBuf, transitionMatricesBuf, stateCountsBuf, biasedConfEnergiesBuf) reduction(+:a) collapse(2)
        for (StateIndex k = 0; k < nThermStates_; ++k) {
            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                if (stateCountsBuf(k, i) > 0) {
                    a += (stateCountsBuf(k, i) + prior()) * biasedConfEnergiesBuf(k, i);
                }
                for (StateIndex j = 0; j < nMarkovStates_; ++j) {
                    auto CKij = transitionCountsBuf(k, i, j);
                    if (CKij > 0) {
                        if (i == j) {
                            a += (static_cast<dtype>(CKij) + prior()) * std::log(transitionMatricesBuf(k, i, j));
                        } else {
                            a += CKij * std::log(transitionMatricesBuf(k, i, j));
                        }
                    }
                }
            }
        }

        /* \sum_{i,k}N_{i}^{(k)}f_{i}^{(k)} */
        // dtype b = 0;
        /*for (decltype(nThermStates_) k = 0; k < nThermStates_; ++k) {
            for (decltype(nMarkovStates_) i = 0; i < nMarkovStates_; ++i) {
                if (stateCountsBuf(k, i) > 0) {
                    b += (stateCountsBuf(k, i) + prior()) * biasedConfEnergiesBuf(k, i);
                }
            }
        }*/
        return a /*+ b*/;
    }

    void updateTransitionMatrices() {
        auto biasedConfEnergiesbuf = biasedConfEnergies_.template unchecked<2>();
        auto lagrangianMultLogBuf = lagrangianMultLog_.firstBuf();

        auto transitionCountsBuf = input_->transitionCounts();
        auto transitionMatricesBuf = transitionMatrices_.template mutable_unchecked<3>();

        for (StateIndex k = 0; k < nThermStates_; ++k) {
            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                scratchM_[i] = 0.0;
                for (StateIndex j = 0; j < nMarkovStates_; ++j) {
                    transitionMatricesBuf(k, i, j) = 0.0;
                    auto C = transitionCountsBuf(k, i, j) + transitionCountsBuf(k, j, i);

                    if (C > 0) { // skip if there were no transitions
                        if (i == j) {
                            // special case: diagonal element
                            transitionMatricesBuf(k, i, j) = 0.5 * C * exp(-lagrangianMultLogBuf(k, i));
                        } else {
                            // regular case
                            auto divisor = numeric::kahan::logsumexp_pair(
                                    lagrangianMultLogBuf(k, j) - biasedConfEnergiesbuf(k, i),
                                    lagrangianMultLogBuf(k, i) - biasedConfEnergiesbuf(k, j));
                            transitionMatricesBuf(k, i, j) = C * exp(-(biasedConfEnergiesbuf(k, j) + divisor));
                        }
                        scratchM_[i] += transitionMatricesBuf(k, i, j);
                    }
                }
            }
            // normalize transition matrix
            dtype maxSum = 0;
            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                if (scratchM_[i] > maxSum) {
                    maxSum = scratchM_[i];
                }
            }
            if (maxSum == 0) maxSum = 1.0; // completely empty T matrix -> generate Id matrix

            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                for (StateIndex j = 0; j < nMarkovStates_; ++j) {
                    if (i == j) {
                        transitionMatricesBuf(k, i, i) = (transitionMatricesBuf(k, i, i) + maxSum - scratchM_[i]) / maxSum;
                        if (0 == transitionMatricesBuf(k, i, i) && 0 < transitionCountsBuf(k, i, i))
                            fprintf(stderr, "# Warning: zero diagonal element T[%d,%d] with non-zero counts.\n",
                                    static_cast<int>(i),
                                    static_cast<int>(i));
                    } else {
                        transitionMatricesBuf(k, i, j) = transitionMatricesBuf(k, i, j) / maxSum;
                    }
                }
            }
        }
    }

    np_array_nfc<dtype> getSampleWeightsForTrajectory(std::size_t trajectoryIndex, std::int32_t thermStateIndex) {
        // k = -1 for unbiased sample weigths.
        auto sampleWeights = np_array_nfc<dtype>(std::vector{input_->dtraj(trajectoryIndex).size()});
        auto sampleWeightsBuf = sampleWeights.template mutable_unchecked<1>();

        auto dtrajBuf = input_->dtraj(trajectoryIndex);
        auto biasMatrixBuf = input_->biasMatrix(trajectoryIndex);

        auto thermStateEnergiesBuf= thermStateEnergies_.firstBuf();
        auto modifiedStateCountsLogBuf = modifiedStateCountsLog_.template unchecked<2>();

        for (auto x = 0; x < input_->sequenceLength(trajectoryIndex); ++x) {
            auto i = dtrajBuf(x);
            if (i < 0) {
                sampleWeightsBuf(x) = inf;
                continue;
            }
            auto o = 0;
            for (StateIndex l = 0; l < nThermStates_; ++l) {
                if (modifiedStateCountsLogBuf(l, i) > -inf) {
                    scratchT_[o++] = modifiedStateCountsLogBuf(l, i) - biasMatrixBuf(x, l);
                }
            }
            auto log_divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratchT_.get(), o);
            if (thermStateIndex == -1) {// get unbiased sample weight
                sampleWeightsBuf(x) = log_divisor;
            } else { // get biased sample weight for given thermState index
                sampleWeightsBuf(x) = biasMatrixBuf(x, thermStateIndex) + log_divisor
                                      - thermStateEnergiesBuf(thermStateIndex);
            }
        }
        return sampleWeights;
    }

};
}
