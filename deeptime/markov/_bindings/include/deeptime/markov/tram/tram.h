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
class TRAMInput : public std::enable_shared_from_this<TRAMInput<dtype>> {

public:
    using DTraj = np_array<std::int32_t>;
    using DTrajs = std::vector<DTraj>;

    using BiasMatrix = np_array_nfc<dtype>;
    using BiasMatrices = std::vector<BiasMatrix>;

    TRAMInput(np_array_nfc<std::int32_t> &&stateCounts, np_array_nfc<std::int32_t> &&transitionCounts,
              DTrajs dtrajs, BiasMatrices biasMatrices)
            : _stateCounts(std::move(stateCounts)),
              _transitionCounts(std::move(transitionCounts)),
              _dtrajs(std::move(dtrajs)),
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

        if (_dtrajs.size() != _biasMatrices.size()) {
            std::stringstream ss;
            ss << "Input invalid. Number of trajectories should be equal to the size of the first dimension "
                  "of the bias matrix.";
            ss << "\nNumber of trajectories: " << _dtrajs.size() << "\nNumber of bias matrices: "
               << _biasMatrices.size();
            throw std::runtime_error(ss.str());
        }
        detail::throwIfInvalid(_stateCounts.shape(0) == _transitionCounts.shape(0),
                               "stateCounts.shape(0) should equal transitionCounts.shape(0)");
        detail::throwIfInvalid(_stateCounts.shape(1) == _transitionCounts.shape(1),
                               "stateCounts.shape(1) should equal transitionCounts.shape(1)");
        detail::throwIfInvalid(_transitionCounts.shape(1) == _transitionCounts.shape(2),
                               "transitionCounts.shape(1) should equal transitionCounts.shape(2)");

        for (std::size_t k = 0; k < _dtrajs.size(); ++k) {
            const auto &dtrajsK = _dtrajs.at(k);
            const auto &biasMatrixK = _biasMatrices.at(k);

            detail::throwIfInvalid(dtrajsK.ndim() == 1,
                                   "dtraj at index {i} has an incorrect number of dimension. ndims should be 1.");
            detail::throwIfInvalid(biasMatrixK.ndim() == 2,
                                   "biasMatrix at index {i} has an incorrect number of dimension. ndims should be 2.");
            detail::throwIfInvalid(dtrajsK.shape(0) == biasMatrixK.shape(0),
                                   "dtraj and biasMatrix at index {i} should be of equal length.");
            detail::throwIfInvalid(biasMatrixK.shape(1) == _transitionCounts.shape(0),
                                   "biasMatrix{i}.shape[1] should be equal to transitionCounts.shape[0].");
        }
    }

    auto biasMatrix(std::size_t k) const {
        return _biasMatrices.at(k).template unchecked<2>();
    }

    auto dtraj(std::size_t k) const {
        return _dtrajs[k].template unchecked<1>();
    }

    auto transitionCounts() const {
        return _transitionCounts.template unchecked<3>();
    }

    auto stateCounts() const {
        return _stateCounts.template unchecked<2>();
    }

    auto sequenceLength(std::size_t k) const {
        return _dtrajs[k].size();
    }

    auto nTrajectories() const {
        return _dtrajs.size();
    };

private:
    np_array_nfc<std::int32_t> _stateCounts;
    np_array_nfc<std::int32_t> _transitionCounts;
    DTrajs _dtrajs;
    BiasMatrices _biasMatrices;
};


template<typename dtype>
struct TRAM {
public:

    TRAM(std::shared_ptr<TRAMInput<dtype>> tramInput, std::size_t callbackInterval)
            : input(tramInput),
              nThermStates(tramInput->stateCounts().shape(0)),
              nMarkovStates(tramInput->stateCounts().shape(1)),
              callbackInterval(callbackInterval),
              biasedConfEnergies(detail::generateFilledArray<dtype>({nThermStates, nMarkovStates}, 0.)),
              lagrangianMultLog(ExchangeableArray<dtype, 2>({nThermStates, nMarkovStates}, 0.)),
              modifiedStateCountsLog(detail::generateFilledArray<dtype>({nThermStates, nMarkovStates}, 0.)),
              thermStateEnergies(ExchangeableArray<dtype, 1>(std::vector<decltype(nThermStates)>{nThermStates}, 0)),
              markovStateEnergies(np_array_nfc<dtype>(std::vector<decltype(nMarkovStates)>{nMarkovStates})),
              transitionMatrices(np_array_nfc<dtype>({nThermStates, nMarkovStates, nMarkovStates})),
              scratchM(std::unique_ptr<dtype[]>(new dtype[nMarkovStates])),
              scratchT(std::unique_ptr<dtype[]>(new dtype[nThermStates])) {
        initLagrangianMult();
    }

    auto energiesPerThermodynamicState() const {
        return *thermStateEnergies.first();
    }

    auto getEnergiesPerMarkovState() const {
        return markovStateEnergies;
    }

    auto getBiasedConfEnergies() const {
        return biasedConfEnergies;
    }

    auto getTransitionMatrices() {
        return transitionMatrices;
    }

    // compute the log-likelihood of observing the input data under the current
    // biasedConfEnergies.
    dtype computeLogLikelihood() {

        dtype logLikelihood = 0.;

        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            logLikelihood += computeSampleLikelihood(K, input->sequenceLength(K));
        }
        logLikelihood += computeTransitionLikelihood();

        return logLikelihood;
    }


    // estimation loop to estimate the free energies. The current iteration number and iteration error are
    // returned through the callback function if it is provided. log-likelihoods are returned if
    // trackLogLikelihoods = true. By default, this is false, since calculating the log-likelihood required
    // an estimation of the transition matrices, which slows down estimation.
    void estimate(std::size_t maxIter, dtype maxErr, bool trackLogLikelihoods = false,
                  const py::object *callback = nullptr) {

        dtype iterationError = 0.;
        dtype logLikelihood = 0.;
        // Array to keep track of _statVectors(K, i) = exp(_thermStateEnergies(K) - _biasedConfEnergies(K, i))
        // difference between previous and current statvectors is used for calculting the iteration error.
        auto statVectors = ExchangeableArray<dtype, 2>(std::vector({nThermStates, nMarkovStates}), 0.);

        for (decltype(maxIter) iterationCount = 0; iterationCount < maxIter; ++iterationCount) {

            // Self-consistent update of the TRAM equations.
            updateLagrangianMult();
            updateStateCounts();
            updateBiasedConfEnergies();

            // Tracking of energy vectors for error calculation.
            updateThermStateEnergies();
            updateStatVectors(statVectors);

            // compare new thermStateEnergies and statVectors with old to get the
            // iteration error (= how much the energies changed).
            iterationError = getError(statVectors);

            if (trackLogLikelihoods) {
                logLikelihood = computeLogLikelihood();
            }

            if (callback != nullptr && iterationCount % callbackInterval == 0) {
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

        // And compute final transition matrices
        computeTransitionMatrices();

        if (iterationError >= maxErr) {
            // We exceeded maxIter but we did not converge.
            std::cout << "TRAM did not converge. Last increment = " << iterationError << std::endl;
        }
    }

    std::vector<np_array_nfc<dtype>> getSampleWeights(std::int32_t K) {
        std::vector<np_array_nfc<dtype>> sampleWeights(nThermStates);

        for (std::size_t i = 0; i < input->nTrajectories(); ++i) {
            sampleWeights.push_back(getSampleWeightsForTrajectory(i, K));
        }
        return sampleWeights;
    }

private:
    std::shared_ptr<TRAMInput<dtype>> input;

    py::ssize_t nThermStates;
    py::ssize_t nMarkovStates;

    std::size_t callbackInterval;

    np_array_nfc<dtype> biasedConfEnergies;
    ExchangeableArray<dtype, 2> lagrangianMultLog;
    np_array_nfc<dtype> modifiedStateCountsLog;

    ExchangeableArray<dtype, 1> thermStateEnergies;
    np_array_nfc<dtype> markovStateEnergies;
    np_array_nfc<dtype> transitionMatrices;

    // scratch matrices used to facilitate calculation of logsumexp
    std::unique_ptr<dtype[]> scratchM;
    std::unique_ptr<dtype[]> scratchT;


    dtype inf = std::numeric_limits<dtype>::infinity();


    constexpr static dtype prior() { return 0.0; }

    constexpr static dtype logPrior() { return 1.0; }

    void initLagrangianMult() {
        auto _transitionCounts = input->transitionCounts();
        auto _lagrangianMultLog = lagrangianMultLog.firstBuf();

        for (decltype(nThermStates) k = 0; k < nThermStates; ++k) {
            for (std::size_t i = 0; i < nMarkovStates; ++i) {
                dtype sum = 0.0;
                for (decltype(nMarkovStates) j = 0; j < nMarkovStates; ++j) {
                    sum += (_transitionCounts(k, i, j) + _transitionCounts(k, j, i));
                }
                _lagrangianMultLog(k, i) = std::log(sum / 2);
            }
        }
    }

    void updateLagrangianMult() {
        lagrangianMultLog.exchange();
        auto _oldLagrangianMultLog = lagrangianMultLog.secondBuf();
        auto _newLagrangianMultLog = lagrangianMultLog.firstBuf();

        auto _biasedConfEnergies = biasedConfEnergies.template unchecked<2>();

        auto _transitionCounts = input->transitionCounts();
        auto _stateCounts = input->stateCounts();

        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
                if (0 == _stateCounts(K, i)) {
                    _newLagrangianMultLog(K, i) = -inf;
                    continue;
                }
                std::size_t o = 0;
                
                for (decltype(nMarkovStates) j = 0; j < nMarkovStates; ++j) {
                    auto CKij = _transitionCounts(K, i, j);
                    // special case: most variables cancel out, here
                    if (i == j) {
                        scratchM[o++] = (0 == CKij) ?
                                        logPrior() : std::log(prior() + (dtype) CKij);
                    } else {
                        auto CK = CKij + _transitionCounts(K, j, i);
                        // todo why?
                        if (0 != CK) {
                            auto divisor = numeric::kahan::logsumexp_pair<dtype>(
                                    _oldLagrangianMultLog(K, j) - _biasedConfEnergies(K, i)
                                    - _oldLagrangianMultLog(K, i) + _biasedConfEnergies(K, j), 0.0);
                            scratchM[o++] = std::log((dtype) CK) - divisor;
                        }
                    }
                }
                _newLagrangianMultLog(K, i) = numeric::kahan::logsumexp_sort_kahan_inplace(scratchM.get(), o);
            }
        }
    }

    // update all biased confirmation energies by looping over all trajectories.
    void updateBiasedConfEnergies() {

        std::fill(biasedConfEnergies.mutable_data(), biasedConfEnergies.mutable_data() + nMarkovStates * nThermStates,
                  inf);

        for (std::size_t i = 0; i < input->nTrajectories(); ++i) {
            updateBiasedConfEnergies(i, input->sequenceLength(i));
        }
    }

    // update conformation energies based on one observed trajectory
    void updateBiasedConfEnergies(std::size_t thermState, std::size_t trajLength) {
        auto _biasMatrix = input->biasMatrix(thermState);
        auto _dtraj = input->dtraj(thermState);

        auto _biasedConfEnergies = biasedConfEnergies.template mutable_unchecked<2>();
        auto _modifiedStateCountsLog = modifiedStateCountsLog.template unchecked<2>();

        dtype divisor {};

        auto* scratch = scratchT.get();

        /* assume that new_biased_conf_energies have been set to INF by the caller in the first call */
        for (std::size_t x = 0; x < trajLength; ++x) {
            std::int32_t i = _dtraj(x);
            if (i < 0) continue; /* skip frames that have negative Markov state indices */
            std::size_t o = 0;
            for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {

                /* applying Hao's speed-up recomendation */
                if (-inf == _modifiedStateCountsLog(K, i)) continue;
                scratch[o++] = _modifiedStateCountsLog(K, i) - _biasMatrix(x, K);
            }
            divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratch, o);

            for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
                _biasedConfEnergies(K, i) = -numeric::kahan::logsumexp_pair(
                        -_biasedConfEnergies(K, i), -(divisor + _biasMatrix(x, K)));
            }

        }
    }


//    template<typename dtype, bool trammbar = false>
    void updateStateCounts() {
        auto _biasedConfEnergies = biasedConfEnergies.template unchecked<2>();
        auto _lagrangianMultLog = lagrangianMultLog.firstBuf();
        auto _modifiedStateCountsLog = modifiedStateCountsLog.template mutable_unchecked<2>();

        auto _stateCounts = input->stateCounts();
        auto _transitionCounts = input->transitionCounts();

        std::size_t o;
        std::int32_t Ci, CK, CKij, CKji, NC;
        dtype divisor, extraStateCounts;

        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
                if (0 == _stateCounts(K, i)) /* applying Hao's speed-up recomendation */
                {
                    _modifiedStateCountsLog(K, i) = -inf;
                    continue;
                }
                Ci = 0;
                o = 0;
                for (decltype(nMarkovStates) j = 0; j < nMarkovStates; ++j) {
                    CKij = _transitionCounts(K, i, j);
                    CKji = _transitionCounts(K, j, i);
                    Ci += CKji;
                    /* special case: most variables cancel out, here */
                    if (i == j) {
                        scratchM[o] = (0 == CKij) ? logPrior() : std::log(
                                prior() + (dtype) CKij);
                        scratchM[o++] += _biasedConfEnergies(K, i);
                        continue;
                    }
                    CK = CKij + CKji;
                    /* special case */
                    if (0 == CK) continue;
                    /* regular case */
                    divisor = numeric::kahan::logsumexp_pair(
                            _lagrangianMultLog(K, j) - _biasedConfEnergies(K, i),
                            _lagrangianMultLog(K, i) - _biasedConfEnergies(K, j));
                    scratchM[o++] = std::log((dtype) CK) + _lagrangianMultLog(K, j) - divisor;
                }
                NC = _stateCounts(K, i) - Ci;
                extraStateCounts = (0 < NC) ? std::log((dtype) NC) + _biasedConfEnergies(K, i) : -inf; /* IGNORE PRIOR */
                _modifiedStateCountsLog(K, i) = numeric::kahan::logsumexp_pair(
                        numeric::kahan::logsumexp_sort_kahan_inplace(scratchM.get(), o), extraStateCounts);
            }
        }

//        if constexpr(trammbar) {
        // todo
        /*if(equilibrium_therm_state_counts && thermStateEnergies)
        {
            for(K=0; K<nThermStates; ++K)
            {
                KM = K * n_conf_states;
                for(i=0; i<n_conf_states; ++i)
                    log_R_K_i[KM + i] += std::log(overcounting_factor);
            }
            for(K=0; K<nThermStates; ++K)
            {
                if(0 < equilibrium_therm_state_counts[K])
                {
                    KM = K * n_conf_states;
                    for(i=0; i<n_conf_states; ++i)
                    {
                        Ki = KM + i;
                        log_R_K_i[Ki] = logsumexp_pair(log_R_K_i[Ki], std::log(equilibrium_therm_state_counts[K]) + thermStateEnergies[K]);
                    }
                }
            }
        } */
//        }

    }

    // Get the error in the energies between this iteration and the previous one.
    dtype getError(ExchangeableArray<dtype, 2> &statVectors) {
        auto _thermEnergies = thermStateEnergies.firstBuf();
        auto _oldThermEnergies = thermStateEnergies.secondBuf();
        auto _statVectors = statVectors.firstBuf();
        auto _oldStatVectors = statVectors.secondBuf();

        dtype maxError = 0;
        dtype energyDelta;

        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            energyDelta = std::abs(_thermEnergies(K) - _oldThermEnergies(K));
            if (energyDelta > maxError) maxError = energyDelta;

            for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
                energyDelta = std::abs(_statVectors(K, i) - _oldStatVectors(K, i));
                if (energyDelta > maxError) maxError = energyDelta;
            }
        }
        return maxError;
    }

    void updateStatVectors(ExchangeableArray<dtype, 2> &statVectors) {
        // move current values to old
        statVectors.exchange();

        // compute new values
        auto _statVectors = statVectors.firstBuf();
        auto _thermStateEnergies = thermStateEnergies.firstBuf();
        auto _biasedConfEnergies = biasedConfEnergies.template unchecked<2>();

        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
                _statVectors(K, i) = std::exp(_thermStateEnergies(K) - _biasedConfEnergies(K, i));
            }
        }
    }

    void computeMarkovStateEnergies() {
        auto _markovStateEnergies = markovStateEnergies.template mutable_unchecked<1>();

        // first reset all confirmation energies to infinity
        for (decltype(nMarkovStates) i = 0; i < nMarkovStates; i++) {
            _markovStateEnergies(i) = inf;
        }

        for (std::size_t i = 0; i < input->nTrajectories(); ++i) {
            auto _dtraj_K = input->dtraj(i);
            auto _biasMatrix_K = input->biasMatrix(i);
            auto trajLength = input->sequenceLength(i);

            computeMarkovStateEnergiesForSingleTrajectory(_biasMatrix_K, _dtraj_K, trajLength);
        }
    }

    template<typename BiasMatrixK, typename Dtraj>
    void
    computeMarkovStateEnergiesForSingleTrajectory(const BiasMatrixK &_biasMatrix_K,
                                                  const Dtraj &_dtraj,
                                                  std::size_t trajlength /*todo this is just dtraj length*/) {
        auto _modifiedStateCountsLog = modifiedStateCountsLog.template unchecked<2>();
        auto _markovStateEnergies = markovStateEnergies.template mutable_unchecked<1>();

        dtype divisor;
        /* assume that markovStateEnergies was set to INF by the caller on the first call */
        for (decltype(trajlength) x = 0; x < trajlength; ++x) {
            std::int32_t i = _dtraj(x);
            if (i < 0) continue;
            std::size_t o = 0;
            for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
                if (-inf == _modifiedStateCountsLog(K, i)) continue;
                scratchT[o++] =
                        _modifiedStateCountsLog(K, i) - _biasMatrix_K(x, K);
            }
            divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratchT.get(), o);
            _markovStateEnergies(i) = -numeric::kahan::logsumexp_pair(-_markovStateEnergies(i), -divisor);
        }
    }

    void updateThermStateEnergies() {
        // move current values to old
        thermStateEnergies.exchange();

        // compute new
        auto _biasedConfEnergies = biasedConfEnergies.template unchecked<2>();
        auto _thermStateEnergies = thermStateEnergies.firstBuf();

        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
                scratchM[i] = -_biasedConfEnergies(K, i);
            }
            _thermStateEnergies(K) = -numeric::kahan::logsumexp_sort_kahan_inplace(scratchM.get(), nMarkovStates);
        }
    }

    // Shift all energies by min(biasedConfEnergies) so the energies don't drift to
    // very large values.
    void shiftEnergiesToHaveZeroMinimum() {
        auto _biasedConfEnergies = biasedConfEnergies.template mutable_unchecked<2>();
        auto _thermStateEnergies = thermStateEnergies.firstBuf();

        dtype shift = 0;

        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
                if (_biasedConfEnergies(K, i) < shift) {
                    shift = _biasedConfEnergies(K, i);
                }
            }
        }
        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            _thermStateEnergies(K) -= shift;

            for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
                _biasedConfEnergies(K, i) -= shift;
            }
        }
    }

    void normalize() {
        auto _biasedConfEnergies = biasedConfEnergies.template mutable_unchecked<2>();
        auto _markovStateEnergies = markovStateEnergies.template mutable_unchecked<1>();
        auto _thermStateEnergies = thermStateEnergies.firstBuf();

        for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
            scratchM[i] = -_markovStateEnergies(i);
        }
        dtype f0 = -numeric::kahan::logsumexp_sort_kahan_inplace(scratchM.get(), nMarkovStates);

        for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
            _markovStateEnergies(i) -= f0;
            for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
                _biasedConfEnergies(K, i) -= f0;
            }
        }
        for (decltype(nThermStates) k = 0; k < nThermStates; ++k) {
            _thermStateEnergies(k) -= f0;
        }

        // update the state counts because they also include biased conf energies.
        // If this is not done after normalizing, the log likelihood computations will
        // not produce the correct output, due to incorrect values for mu(x).
        updateStateCounts();
    }

// log likelihood of observing a sampled trajectory from the local equilibrium.
// i.e. the sum over all sample likelihoods from one trajectory within on
// thermodynamic state.
dtype computeSampleLikelihood(std::size_t thermState, std::size_t trajLength) {
        auto _modifiedStateCountsLog = modifiedStateCountsLog.template unchecked<2>();
        auto _biasMatrix = input->biasMatrix(thermState);
        auto _dtraj = input->dtraj(thermState);

/* -\sum_{x}\log\sum_{l}R_{i(x)}^{(l)}e^{-b^{(l)}(x)+f_{i(x)}^{(l)}} */
        dtype logLikelihood = 0;
        for (std::size_t x = 0; x < trajLength; ++x) {
            std::int32_t o = 0;
            std::int32_t i = _dtraj(x);
            if (i < 0) continue;
            for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
                if (_modifiedStateCountsLog(K, i) > 0)
                    scratchT[o++] =_modifiedStateCountsLog(K, i) - _biasMatrix(x, K);
            }
            logLikelihood -= numeric::kahan::logsumexp_sort_kahan_inplace(scratchT.get(), o);
        }
        return logLikelihood;
    }


/* TRAM log-likelihood that comes from the terms containing discrete quantities.
 * i.e. the likelihood of observing the observed transitions. */
    dtype computeTransitionLikelihood() {
        auto _biasedConfEnergies = biasedConfEnergies.template unchecked<2>();
        auto _transitionCounts = input->transitionCounts();
        auto _stateCounts = input->stateCounts();
        auto _transitionMatrices = transitionMatrices.template unchecked<3>();

        std::int32_t CKij;

        /* \sum_{i,j,k}c_{ij}^{(k)}\log p_{ij}^{(k)} */
        dtype a = 0;
        computeTransitionMatrices();

        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
                for (decltype(nMarkovStates) j = 0; j < nMarkovStates; ++j) {
                    CKij = _transitionCounts(K, i, j);
                    if (0 == CKij) continue;
                    if (i == j) {
                        a += ((dtype) CKij + prior()) * std::log(_transitionMatrices(K, i, j));
                    } else {
                        a += CKij * std::log(_transitionMatrices(K, i, j));
                    }
                }
            }
        }
        /* \sum_{i,k}N_{i}^{(k)}f_{i}^{(k)} */
        dtype b = 0;
        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
                if (_stateCounts(K, i) > 0)
                    b += (_stateCounts(K, i) + prior()) * _biasedConfEnergies(K, i);
            }
        }
        return a + b;
    }

    void computeTransitionMatrices() {

        auto _biasedConfEnergies = biasedConfEnergies.template unchecked<2>();
        auto _lagrangianMultLog = lagrangianMultLog.firstBuf();

        auto _transitionCounts = input->transitionCounts();
        auto _transitionMatrices = transitionMatrices.template mutable_unchecked<3>();

        std::int32_t C;
        dtype divisor, maxSum;
        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
                scratchM[i] = 0.0;
                for (decltype(nMarkovStates) j = 0; j < nMarkovStates; ++j) {
                    _transitionMatrices(K, i, j) = 0.0;
                    C = _transitionCounts(K, i, j) + _transitionCounts(K, j, i);
                    /* special case: this element is zero */
                    if (0 == C) continue;
                    if (i == j) {
                        /* special case: diagonal element */
                        _transitionMatrices(K, i, j) = 0.5 * C * exp(-_lagrangianMultLog(K, i));
                    } else {
                        /* regular case */
                        divisor = numeric::kahan::logsumexp_pair(
                                _lagrangianMultLog(K, j) - _biasedConfEnergies(K, i),
                                _lagrangianMultLog(K, i) - _biasedConfEnergies(K, j));
                        _transitionMatrices(K, i, j) = C * exp(-(_biasedConfEnergies(K, j) + divisor));
                    }
                    scratchM[i] += _transitionMatrices(K, i, j);
                }
            }
            /* normalize T matrix */ /* TODO: unify with util._renormalize_transition_matrix? */
            maxSum = 0;
            for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) if (scratchM[i] > maxSum) maxSum = scratchM[i];
            if (maxSum == 0) maxSum = 1.0; /* completely empty T matrix -> generate Id matrix */
            for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
                for (decltype(nMarkovStates) j = 0; j < nMarkovStates; ++j) {
                    if (i == j) {
                        _transitionMatrices(K, i, i) =
                                (_transitionMatrices(K, i, i) + maxSum - scratchM[i]) / maxSum;
                        if (0 == _transitionMatrices(K, i, i) && 0 < _transitionCounts(K, i, i))
                            fprintf(stderr, "# Warning: zero diagonal element T[%d,%d] with non-zero counts.\n",
                                    static_cast<int>(i),
                                    static_cast<int>(i));
                    } else {
                        _transitionMatrices(K, i, j) = _transitionMatrices(K, i, j) / maxSum;
                    }
                }
            }
        }
    }

    np_array_nfc<dtype> getSampleWeightsForTrajectory(std::size_t trajectoryIndex, std::int32_t K) {
        // k = -1 for unbiased sample weigths.
        auto sampleWeights = np_array_nfc<dtype>(std::vector{input->dtraj(trajectoryIndex).size()});
        auto _sampleWeights = sampleWeights.template mutable_unchecked<1>();

        auto _dtraj = input->dtraj(trajectoryIndex);
        auto _biasMatrix = input->biasMatrix(trajectoryIndex);

        auto _thermStateEnergies = thermStateEnergies.firstBuf();
        auto _modifiedStateCountsLog = modifiedStateCountsLog.template unchecked<2>();

        int L, o, i;
        dtype log_divisor;

        for (auto x = 0; x < input->sequenceLength(trajectoryIndex); ++x) {
            i = _dtraj(x);
            if (i < 0) {
                _sampleWeights(x) = inf;
                continue;
            }
            o = 0;
            for (L = 0; L < nThermStates; ++L) {
                if (-inf == _modifiedStateCountsLog(L, i)) continue;
                scratchT[o++] = _modifiedStateCountsLog(L, i) - _biasMatrix(x, L);
            }
            log_divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratchT.get(), o);
            if (K == -1)
                _sampleWeights(x) = log_divisor;
            else
                _sampleWeights(x) = _biasMatrix(x, K) + log_divisor - _thermStateEnergies(K);
        }
        return sampleWeights;
    }

};
}
