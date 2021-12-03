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
np_array_nfc<dtype> getFilledArray(std::vector<std::size_t> dims, dtype fillValue) {
    auto array = np_array_nfc<dtype>(dims);

    auto totalSize = std::accumulate(begin(dims), end(dims), 1, std::multiplies<dtype>());
    std::fill(array.mutable_data(), array.mutable_data() + totalSize, fillValue);

    return array;
}

template<py::ssize_t Dims, typename Array>
auto getMutableBuf(Array &&array) {
    return array.template mutable_unchecked<Dims>();
}
}

template<typename dtype, py::ssize_t Dims>
class ExchangeableArray {
    using MutableBufferType = decltype(detail::getMutableBuf<Dims>(std::declval<np_array<dtype>>()));
public:
    template<typename Shape>
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

    auto *second() {
        return current ? &std::get<1>(arrays) : &std::get<0>(arrays);
    }

    auto &firstBuf() {
        return current ? *std::get<0>(buffers) : *std::get<1>(buffers);
    }

    auto &secondBuf() {
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
    using DTraj = np_array_nfc<std::int32_t>;
    using DTrajs = std::vector<DTraj>;

    using BiasMatrix = np_array_nfc<dtype>;
    using BiasMatrices = std::vector<BiasMatrix>;

    TRAMInput(np_array_nfc<std::int32_t> &&stateCounts, np_array_nfc<std::int32_t> &&transitionCounts,
              const DTrajs &dtrajs, const BiasMatrices &biasMatrix)
            : _stateCounts(std::move(stateCounts)),
              _transitionCounts(std::move(transitionCounts)),
              _dtrajs(dtrajs),
              _biasMatrices(biasMatrix) {
        validateInput();
    }

    TRAMInput() = default;

    TRAMInput(const TRAMInput &) = delete;

    TRAMInput &operator=(const TRAMInput &) = delete;

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

        for (std::size_t K = 0; K < _dtrajs.size(); ++K) {
            const auto &dtrajs_K = _dtrajs.at(K);
            const auto &biasMatrix_K = _biasMatrices.at(K);

            detail::throwIfInvalid(dtrajs_K.ndim() == 1,
                                   "dtraj at index {i} has an incorrect number of dimension. ndims should be 1.");
            detail::throwIfInvalid(biasMatrix_K.ndim() == 2,
                                   "biasMatrix at index {i} has an incorrect number of dimension. ndims should be 2.");
            detail::throwIfInvalid(dtrajs_K.shape(0) == biasMatrix_K.shape(0),
                                   "dtraj and biasMatrix at index {i} should be of equal length.");
            detail::throwIfInvalid(biasMatrix_K.shape(1) == _transitionCounts.shape(0),
                                   "biasMatrix{i}.shape[1] should be equal to transitionCounts.shape[0].");
        }
    }

    auto biasMatrix(std::size_t K) const {
        return _biasMatrices.at(K).template unchecked<2>();
    }

    auto dtraj(std::size_t K) const {
        return _dtrajs.at(K).template unchecked<1>();
    }

    auto transitionCounts() const {
        return _transitionCounts.template unchecked<3>();
    }

    auto stateCounts() const {
        return _stateCounts.template unchecked<2>();
    }

    auto sequenceLength(std::size_t K) const {
        return _dtrajs.at(K).size();
    }

private:
    np_array_nfc<std::int32_t> _stateCounts;
    np_array_nfc<std::int32_t> _transitionCounts;
    DTrajs _dtrajs;
    BiasMatrices _biasMatrices;
};


template<typename dtype>
struct TRAM {
public:

    TRAM(std::shared_ptr<TRAMInput<dtype>> &tramInput, std::size_t callbackInterval)
            : input(std::shared_ptr(tramInput)),
              nThermStates(tramInput->stateCounts().shape(0)),
              nMarkovStates(tramInput->stateCounts().shape(1)),
              callbackInterval(callbackInterval),
              biasedConfEnergies(detail::getFilledArray<dtype>({nThermStates, nMarkovStates}, 0.)),
              lagrangianMultLog(ExchangeableArray<dtype, 2>(std::vector({nThermStates, nMarkovStates}), 0.)),
              modifiedStateCountsLog(detail::getFilledArray<dtype>({nThermStates, nMarkovStates}, 0.)),
              thermStateEnergies(ExchangeableArray<dtype, 1>(std::vector<decltype(nThermStates)>{nThermStates}, 0)),
              markovStateEnergies(np_array_nfc<dtype>(std::vector<decltype(nMarkovStates)>{nMarkovStates})),
              transitionMatrices(np_array_nfc<dtype>({nThermStates, nMarkovStates, nMarkovStates})),
              scratchM(std::unique_ptr<dtype[]>(new dtype[nMarkovStates])),
              scratchT(std::unique_ptr<dtype[]>(new dtype[nThermStates])) {

        initLagrangianMult();
    }

    auto getEnergiesPerThermodynamicState() {
        return *thermStateEnergies.first();
    }

    auto getEnergiesPerMarkovState() const {
        return markovStateEnergies;
    }

    auto getBiasedConfEnergies() const {
        return biasedConfEnergies;
    }

    auto getTransitionMatrices() {
        computeTransitionMatrices();
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

	    std::cout<<iterationError << " " << logLikelihood << std::endl;
            if (callback != nullptr && iterationCount % callbackInterval == 0) {
                // TODO: callback doesn't work in release???
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

        if (iterationError >= maxErr) {
            // We exceeded maxIter but we did not converge.
            std::cout << "TRAM did not converge. Last increment = " << iterationError << std::endl;
	    std::cout << "current LL: " << logLikelihood << std::endl;
        }
    }


private:
    std::shared_ptr<TRAMInput<dtype>> input;

    std::size_t nThermStates;
    std::size_t nMarkovStates;

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


// TODO: WTF to do with this????
    dtype TRAM_PRIOR() { return 0.0; }

    dtype TRAM_LOG_PRIOR() { return 1.0; }

    void initLagrangianMult() {
        auto _transitionCounts = input->transitionCounts();
        auto _lagrangianMultLog = lagrangianMultLog.firstBuf();

        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            for (std::size_t i = 0; i < nMarkovStates; ++i) {
                dtype sum = 0.0;

                for (decltype(nMarkovStates) j = 0; j < nMarkovStates; ++j) {
                    sum += (_transitionCounts(K, i, j) +
                            _transitionCounts(K, j, i));
                }
                _lagrangianMultLog(K, i) = std::log(sum / 2);
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

        std::int32_t CK, CKij;
        dtype divisor;

        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
                if (0 == _stateCounts(K, i)) {
                    _newLagrangianMultLog(K, i) = -inf;
                    continue;
                }
                std::size_t o = 0;
                for (decltype(nMarkovStates) j = 0; j < nMarkovStates; ++j) {
                    CKij = _transitionCounts(K, i, j);
                    /* special case: most variables cancel out, here */
                    if (i == j) {
                        scratchM[o++] = (0 == CKij) ?
                                        TRAM_LOG_PRIOR() : log(TRAM_PRIOR() + (dtype) CKij);
                        continue;
                    }
                    CK = CKij + _transitionCounts(K, j, i);
                    /* special case */
                    if (0 == CK) continue;
                    /* regular case */
                    divisor = numeric::kahan::logsumexp_pair<dtype>(
                            _oldLagrangianMultLog(K, j) - _biasedConfEnergies(K, i)
                            - _oldLagrangianMultLog(K, i) + _biasedConfEnergies(K, j), 0.0);
                    scratchM[o++] = log((dtype) CK) - divisor;
                }
                _newLagrangianMultLog(K, i) = numeric::kahan::logsumexp_sort_kahan_inplace(scratchM.get(),
                                                                                           scratchM.get() + o);
            }
        }
    }

    // update all biased confirmation energies by looping over all trajectories.
    void updateBiasedConfEnergies() {

        std::fill(biasedConfEnergies.mutable_data(), biasedConfEnergies.mutable_data() + nMarkovStates * nThermStates,
                  inf);

        for (decltype(nThermStates) K = 0; K < nThermStates; K++) {
            updateBiasedConfEnergies(K, input->sequenceLength(K));
        }
    }

    // update conformation energies based on one observed trajectory
    void updateBiasedConfEnergies(std::size_t thermState, std::size_t trajLength) {
        auto _biasMatrix = input->biasMatrix(thermState);
        auto _dtraj = input->dtraj(thermState);

        auto _biasedConfEnergies = biasedConfEnergies.template mutable_unchecked<2>();
        auto _modifiedStateCountsLog = modifiedStateCountsLog.template unchecked<2>();

        dtype divisor = 0;

        /* assume that new_biased_conf_energies have been set to INF by the caller in the first call */
        for (std::size_t x = 0; x < trajLength; ++x) {
            std::int32_t i = _dtraj(x);
            if (i < 0) continue; /* skip frames that have negative Markov state indices */
            std::size_t o = 0;
            for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {

                /* applying Hao's speed-up recomendation */
                if (-inf == _modifiedStateCountsLog(K, i)) continue;
                scratchT[o++] = _modifiedStateCountsLog(K, i) - _biasMatrix(x, K);
            }
            divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratchT.get(), o);

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
                        scratchM[o] = (0 == CKij) ? TRAM_LOG_PRIOR() : log(
                                TRAM_PRIOR() + (dtype) CKij);
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
                    scratchM[o++] = log((dtype) CK) + _lagrangianMultLog(K, j) - divisor;
                }
                NC = _stateCounts(K, i) - Ci;
                extraStateCounts = (0 < NC) ? log((dtype) NC) + _biasedConfEnergies(K, i)
                                            : -inf; /* IGNORE PRIOR */
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
                    log_R_K_i[KM + i] += log(overcounting_factor);
            }
            for(K=0; K<nThermStates; ++K)
            {
                if(0 < equilibrium_therm_state_counts[K])
                {
                    KM = K * n_conf_states;
                    for(i=0; i<n_conf_states; ++i)
                    {
                        Ki = KM + i;
                        log_R_K_i[Ki] = logsumexp_pair(log_R_K_i[Ki], log(equilibrium_therm_state_counts[K]) + thermStateEnergies[K]);
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

        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            auto _dtraj_K = input->dtraj(K);
            auto _biasMatrix_K = input->biasMatrix(K);
            auto trajLength = input->sequenceLength(K);

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
            _thermStateEnergies(i) -= f0;
            for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
                _biasedConfEnergies(K, i) -= f0;
            }
        }
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
                    scratchT[o++] = _modifiedStateCountsLog(K, i) - _biasMatrix(x, K);
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
                        a += ((dtype) CKij + TRAM_PRIOR()) * log(_transitionMatrices(K, i, j));
                    } else {
                        a += CKij * log(_transitionMatrices(K, i, j));
                    }
                }
            }
        }
        /* \sum_{i,k}N_{i}^{(k)}f_{i}^{(k)} */
        dtype b = 0;
        for (decltype(nThermStates) K = 0; K < nThermStates; ++K) {
            for (decltype(nMarkovStates) i = 0; i < nMarkovStates; ++i) {
                if (_stateCounts(K, i) > 0)
                    b += (_stateCounts(K, i) + TRAM_PRIOR()) * _biasedConfEnergies(K, i);
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
//    void get_pointwise_unbiased_free_energies(int K) {
//        auto _dtrajs = input.dtraj();
//        auto _bias_matrix = input.biasMatrix();
//
//        auto _therm_energies = thermStateEnergies.template unchecked<1>();
//        auto _modified_state_counts_log = modifiedStateCountsLog.template unchecked<2>();
//
//        auto _scratch_T = scratchT.template unchecked<1>();
//
//        int traj_length = _dtraj.shape[0];
//        np_array<dtype> pointwise_unbiased_free_energies = np_array<dtype>({traj_length});
//        auto _pointwise_unbiased_free_energies = pointwise_unbiased_free_energies.template mutable_unchecked<2>();
//
//        int L, o, i, x;
//        dtype log_divisor;
//
//        for (int x = 0; x < traj_length; ++x) {
//            i = _dtraj(x);
//            if (i < 0) {
//                _pointwise_unbiased_free_energies(x) = inf;
//                continue;
//            }
//            o = 0;
//            for (L = 0; L < nThermStates; ++L) {
//                if (-inf == _modified_state_counts_log(L, i)) continue;
//                _scratch_T(o++) =
//                        _modified_state_counts_log(L, i) - _bias_matrix(x, L);
//            }
//            log_divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratchT, o);
//            if (K == -1)
//                pointwise_unbiased_free_energies_ptr[x] = log_divisor;
//            else
//                pointwise_unbiased_free_energies_ptr[x] =
//                        bias_energy_sequence_ptr[x * nThermStates + k] + log_divisor - therm_energies_ptr[k];
//        }
//    }
//};
};


template<typename dtype>
extern dtype
_bar_df(np_array_nfc<dtype> db_IJ, std::int32_t L1, np_array_nfc<dtype> db_JI, std::int32_t L2,
        np_array_nfc<dtype> scratch) {
    py::buffer_info db_IJ_buf = db_IJ.request();
    py::buffer_info db_JI_buf = db_JI.request();
    py::buffer_info scratch_buf = scratch.request();

    auto *db_IJ_ptr = (dtype *) db_IJ_buf.ptr;
    auto *db_JI_ptr = (dtype *) db_JI_buf.ptr;
    auto *scratch_ptr = (dtype *) scratch_buf.ptr;

    std::int32_t i;
    dtype ln_avg1;
    dtype ln_avg2;
    for (i = 0; i < L1; i++) {
        scratch_ptr[i] = db_IJ_ptr[i] > 0 ? 0 : db_IJ_ptr[i];
    }
    ln_avg1 = numeric::kahan::logsumexp_sort_kahan_inplace(scratch_ptr, scratch_ptr + L1);
    for (i = 0; i < L1; i++) {
        scratch_ptr[i] = db_JI_ptr[i] > 0 ? 0 : db_JI_ptr[i];
    }
    ln_avg2 = numeric::kahan::logsumexp_sort_kahan_inplace(scratch_ptr, scratch_ptr + L2);
    return ln_avg2 - ln_avg1;
}
}
