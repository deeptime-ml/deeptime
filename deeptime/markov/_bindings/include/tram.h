//
// Created by Maaike on 15/11/2021.
//

#pragma once

#include <cstdio>
#include <cassert>
#include <utility>
#include "common.h"
#include "kahan_summation.h"

namespace deeptime {
namespace tram {

// TODO: WTF to do with this????
double THERMOTOOLS_TRAM_PRIOR = 0.0;
double THERMOTOOLS_TRAM_LOG_PRIOR = 1.0;


//template<typename dtype>
//struct TwoDimArray {
//
//    auto* column(std::size_t i) {
//        return data.data() + i*width;
//    }
//
//private:
//    std::vector<dtype> data;
//};
//template<typename dtype> using BiasedConfEnergies = TwoDimArray<dtype>;

namespace detail {
void throwIfInvalid(bool isValid, const std::string &message) {
    if (!isValid) {
        throw std::runtime_error(message);
    }
}

template<typename dtype>
np_array_nfc<dtype> getFilledArray(std::vector<std::size_t> dims, dtype fillValue) {
    auto array = np_array_nfc<dtype>(dims);

    auto totalSize = std::accumulate(begin(dims), end(dims), 1, std::multiplies<double>());
    std::fill(array.mutable_data(), array.mutable_data() + totalSize, fillValue);

    return array;
}
}

template<typename dtype>
class TRAMInput : public std::enable_shared_from_this<TRAMInput<dtype>> {
    using DTraj = np_array_nfc<std::int32_t>;
    using DTrajs = std::vector<DTraj>;

    using BiasMatrix = np_array_nfc<dtype>;
    using BiasMatrices = std::vector<BiasMatrix>;

public:
    TRAMInput(np_array_nfc<std::int32_t> &&stateCounts, np_array_nfc<std::int32_t> &&transitionCounts,
              py::list dtrajs, py::list biasMatrix)
            : _stateCounts(std::move(stateCounts)),
              _transitionCounts(std::move(transitionCounts)) {

        _dtrajs.reserve(dtrajs.size());
        std::transform(dtrajs.begin(), dtrajs.end(), std::back_inserter(_dtrajs), [](const auto &pyObject) {
            return py::cast<DTraj>(pyObject);
        });
        _biasMatrices.reserve(biasMatrix.size());
        std::transform(biasMatrix.begin(), biasMatrix.end(), std::back_inserter(_biasMatrices),
                       [](const auto &pyObject) {
                           return py::cast<BiasMatrix>(pyObject);
                       });
        validateInput();
    }

    TRAMInput() = default;

    TRAMInput(const TRAMInput &) = delete;

    TRAMInput &operator=(const TRAMInput &) = delete;

    void validateInput() {

        if (_dtrajs.size() != _biasMatrices.size()) {
            std::stringstream ss;
            ss << "Input invalid. Number of trajectories should be equal to the size of the first dimension "
                  "of the bias matrix.";
            ss << "\nNumber of trajectories: " << _dtrajs.size() << "\nNumber of bias matrices: "
               << _biasMatrices.size();
            throw std::runtime_error(ss.str());
        }

        for (int K = 0; K < _dtrajs.size(); ++K) {
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

    auto biasMatrix(int K) const {
        return _biasMatrices.at(K).template unchecked<2>();
    }

    auto dtraj(int K) const {
        return _dtrajs.at(K).template unchecked<1>();
    }

    auto transitionCounts() const {
        return _transitionCounts.template unchecked<3>();
    }

    auto stateCounts() const {
        return _stateCounts.template unchecked<2>();
    }

    auto sequenceLength(int K) const {
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
    std::size_t nThermStates;
    std::size_t nMarkovStates;

    std::shared_ptr<TRAMInput<dtype>> input;

    // # TODO: wrap vector in something nice for happy indexing
    // TODO: make this a vector (no need for np_array)
    // TODO: find a better name than biasedConfEnergies!
    np_array_nfc<dtype> biasedConfEnergies;
    np_array_nfc<dtype> lagrangianMultLog;
    np_array_nfc<dtype> modifiedStateCountsLog;

//    std::vector<dtype> biasedConfEnergies;
//    std::vector<dtype> lagrangianMultLog;
//    std::vector<dtype> modifiedStateCountsLog;


    np_array_nfc<dtype> markovStateEnergies;
    np_array_nfc<dtype> thermStateEnergies;
    np_array_nfc<dtype> transitionMatrices;
    np_array_nfc<dtype> statVectors;

    std::size_t saveConvergenceInfo;

    // scratch matrices used to facilitate calculation of logsumexp
    std::vector<dtype> scratchM;
    std::vector<dtype> scratchT;

    dtype inf = std::numeric_limits<dtype>::infinity();

    TRAM(std::shared_ptr<TRAMInput<dtype>> &tramInput, std::size_t saveConvergenceInfo)
            : saveConvergenceInfo(saveConvergenceInfo),
              input(std::shared_ptr(tramInput)),
              nThermStates(tramInput->stateCounts().shape(0)),
              nMarkovStates(tramInput->stateCounts().shape(1)),
              scratchM(std::vector<dtype>(nMarkovStates)),
              scratchT(std::vector<dtype>(nMarkovStates)),
              biasedConfEnergies(detail::getFilledArray<dtype>({nThermStates, nMarkovStates}, 0.)),
              lagrangianMultLog(detail::getFilledArray<dtype>({nThermStates, nMarkovStates}, 0.)),
              modifiedStateCountsLog(detail::getFilledArray<dtype>({nThermStates, nMarkovStates}, 0.)),
              transitionMatrices(np_array_nfc<dtype>({nThermStates, nMarkovStates, nMarkovStates})),
              markovStateEnergies(np_array_nfc<dtype>(std::vector<decltype(nMarkovStates)>{nMarkovStates})),
              thermStateEnergies(np_array_nfc<dtype>(std::vector<decltype(nThermStates)>{nThermStates})),
              statVectors(np_array_nfc<dtype>({nThermStates, nMarkovStates})) {

        initLagrangianMult();
    }

    auto getBiasedConfEnergies() {
        return biasedConfEnergies;
    }


    void estimate(int maxIter = 1000, dtype maxErr = 1e-8) {

        int iterationCount = 0;
        dtype iterationError = 0;

//#TODO: do something with these. logging?
//        increments = []
//        log_likelihoods = []


        for (int m = 0; m < maxIter; ++m) {
            iterationCount += 1;

            // Self-consistent update of the TRAM equations.
            updateLagrangianMult();
            updateStateCounts();
            updateBiasedConfEnergies();

            // Save old values of these arrays to use for calculating the iteration error.
            auto bufferInfo = thermStateEnergies.request();
            auto oldThermEnergies = np_array_nfc<dtype>(bufferInfo);

            bufferInfo = statVectors.request();
            auto oldStatVectors = np_array_nfc<dtype>(bufferInfo);

            // Compute their respective new values
            computeThermStateEnergies();
            computeStatVectors();

            // compare new with old to get the iteration error (= how much the energies changed).
            iterationError = getError(oldThermEnergies, oldStatVectors);

//            if (iterationCount == saveConvergenceInfo) {
//                iterationCount = 0;
//                increments.append(iterationError)
//                log_likelihoods.append(l)
//            }
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
        computeThermStateEnergies();
        normalize();

        if (iterationError >= maxErr) {
            // We exceeded maxIter but we did not converge.
            std::cout << "TRAM did not converge. Last increment = " << iterationError;
        }
    }

    void initLagrangianMult() {
        auto _transitionCounts = input->transitionCounts();
        auto _lagrangianMultLog = lagrangianMultLog.template mutable_unchecked<2>();

        for (int K = 0; K < nThermStates; ++K) {
            for (int i = 0; i < nMarkovStates; ++i) {
                dtype sum = 0.0;

                for (int j = 0; j < nMarkovStates; ++j) {
                    sum += (_transitionCounts(K, i, j) +
                            _transitionCounts(K, j, i));
                }
                _lagrangianMultLog(K, i) = std::log(sum / 2);
            }
        }
    }


    void updateLagrangianMult() {
        auto _lagrangianMultLog = lagrangianMultLog.template unchecked<2>();
        auto _biasedConfEnergies = biasedConfEnergies.template unchecked<2>();

        auto _transitionCounts = input->transitionCounts();
        auto _stateCounts = input->stateCounts();

        auto newLagrangianMultLog = np_array_nfc<dtype>({nThermStates, nMarkovStates});
        auto _newLagrangianMultLog = newLagrangianMultLog.template mutable_unchecked<2>();

        int CK, CKij;
        dtype divisor;

        for (int K = 0; K < nThermStates; ++K) {
            for (int i = 0; i < nMarkovStates; ++i) {
                if (0 == _stateCounts(K, i)) {
                    _newLagrangianMultLog(K, i) = -inf;
                    continue;
                }
                std::size_t o = 0;
                for (int j = 0; j < nMarkovStates; ++j) {
                    CKij = _transitionCounts(K, i, j);
                    /* special case: most variables cancel out, here */
                    if (i == j) {
                        scratchM[o++] = (0 == CKij) ?
                                        THERMOTOOLS_TRAM_LOG_PRIOR : log(THERMOTOOLS_TRAM_PRIOR + (dtype) CKij);
                        continue;
                    }
                    CK = CKij + _transitionCounts(K, j, i);
                    /* special case */
                    if (0 == CK) continue;
                    /* regular case */
                    divisor = numeric::kahan::logsumexp_pair(
                            _lagrangianMultLog(K, j) - _biasedConfEnergies(K, i)
                            - _lagrangianMultLog(K, i) + _biasedConfEnergies(K, j), 0.0);
                    scratchM[o++] = log((dtype) CK) - divisor;
                }
                _newLagrangianMultLog(K, i) = numeric::kahan::logsumexp_sort_kahan_inplace(scratchM.begin(),
                                                                                           scratchM.begin() + o);
            }
        }
        lagrangianMultLog = std::move(newLagrangianMultLog);
    }

    dtype updateBiasedConfEnergies(int return_log_l = 1) {
        // TODO: what to do with log L?
        dtype logLikelihood = 0.0;

        std::fill(biasedConfEnergies.mutable_data(), biasedConfEnergies.mutable_data() + nMarkovStates * nThermStates,
                  inf);

        for (int K = 0; K < nThermStates; ++K) {
            logLikelihood += updateBiasedConfEnergies(K, return_log_l, input->sequenceLength(K));
        }
        return logLikelihood;
    }

    dtype updateBiasedConfEnergies(int therm_state, bool returnLogLikelihood, int trajLength) {
        auto _biasedConfEnergies = biasedConfEnergies.template mutable_unchecked<2>();
        auto _modifiedStateCountsLog = modifiedStateCountsLog.template unchecked<2>();

        auto _dtraj = input->dtraj(therm_state);
        auto _biasMatrix = input->biasMatrix(therm_state);

        dtype divisor, logLikelihood = 0;

        /* assume that new_biased_conf_energies have been set to INF by the caller in the first call */
        for (int x = 0; x < trajLength; ++x) {
            int i = _dtraj(x);
            if (i < 0) continue; /* skip frames that have negative Markov state indices */
            int o = 0;
            for (int K = 0; K < nThermStates; ++K) {

                /* applying Hao's speed-up recomendation */
                if (-inf == _modifiedStateCountsLog(K, i)) continue;
                scratchT[o++] = _modifiedStateCountsLog(K, i) - _biasMatrix(x, K);
            }
            divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratchT.begin(), o);

            for (int K = 0; K < nThermStates; ++K) {
                _biasedConfEnergies(K, i) = -numeric::kahan::logsumexp_pair(
                        -_biasedConfEnergies(K, i), //TODO: THIS SHOULD BE INF?????
                        -(divisor + _biasMatrix(x, K)));
            }
        }

        // TODO: mechanism to save this progress indicator. Maybe a callback?
        if (returnLogLikelihood) {
            /* -\sum_{x}\log\sum_{l}R_{i(x)}^{(l)}e^{-b^{(l)}(x)+f_{i(x)}^{(l)}} */
            logLikelihood = 0;
            for (int x = 0; x < trajLength; ++x) {
                int o = 0;
                int i = _dtraj(x);
                if (i < 0) continue;
                for (int K = 0; K < nThermStates; ++K) {
                    if (_modifiedStateCountsLog(K, i) > 0)
                        scratchT[o++] =
                                _modifiedStateCountsLog(K, i) - _biasMatrix(x, K);
                }
                logLikelihood -= numeric::kahan::logsumexp_sort_kahan_inplace(scratchT.begin(), o);
            }
        }
        return logLikelihood;
    }

//    template<typename dtype, bool trammbar = false>
    void updateStateCounts() {
        auto _biasedConfEnergies = biasedConfEnergies.template unchecked<2>();
        auto _lagrangianMultLog = lagrangianMultLog.template unchecked<2>();
        auto _modifiedStateCountsLog = modifiedStateCountsLog.template mutable_unchecked<2>();

        auto _stateCounts = input->stateCounts();
        auto _transitionCounts = input->transitionCounts();

        int o;
        int Ci, CK, CKij, CKji, NC;
        dtype divisor, extraStateCounts;

        for (int K = 0; K < nThermStates; ++K) {
            for (int i = 0; i < nMarkovStates; ++i) {
                if (0 == _stateCounts(K, i)) /* applying Hao's speed-up recomendation */
                {
                    _modifiedStateCountsLog(K, i) = -inf;
                    continue;
                }
                Ci = 0;
                o = 0;
                for (int j = 0; j < nMarkovStates; ++j) {
                    CKij = _transitionCounts(K, i, j);
                    CKji = _transitionCounts(K, j, i);
                    Ci += CKji;
                    /* special case: most variables cancel out, here */
                    if (i == j) {
                        scratchM[o] = (0 == CKij) ? THERMOTOOLS_TRAM_LOG_PRIOR : log(
                                THERMOTOOLS_TRAM_PRIOR + (dtype) CKij);
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
                        numeric::kahan::logsumexp_sort_kahan_inplace(scratchM.begin(), o), extraStateCounts);
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

    dtype getError(np_array_nfc<dtype> &newThermEnergies, np_array_nfc<dtype> &newStatVectors) {
        auto _thermEnergies = thermStateEnergies.template unchecked<1>();
        auto _newThermEnergies = newThermEnergies.template unchecked<1>();
        auto _statVectors = statVectors.template unchecked<2>();
        auto _newStatVectors = newStatVectors.template unchecked<2>();

        dtype maxError = 0;
        dtype deltaThermEnergy;
        dtype deltaStatVector;

        for (int K = 0; K < nThermStates; ++K) {
            auto deltaThermEnergy = std::abs(_thermEnergies(K) - _newThermEnergies(K));
            if (deltaThermEnergy > maxError) maxError = deltaThermEnergy;

            for (int i = 0; i < nThermStates; ++i) {
                deltaStatVector = std::abs(_statVectors(K, i) - _newStatVectors(K, i));
                if (deltaStatVector > maxError) maxError = deltaStatVector;
            }
        }
        return maxError;
    }

    np_array_nfc<dtype> computeStatVectors() {
        auto statVectors = np_array_nfc<dtype>({nThermStates, nMarkovStates});
        auto _statVectors = statVectors.template mutable_unchecked<2>();
        auto _thermStateEnergies = thermStateEnergies.template unchecked<1>();
        auto _biasedConfEnergies = biasedConfEnergies.template unchecked<2>();

        for (int K = 0; K < nThermStates; ++K) {
            for (int i = 0; i < nMarkovStates; ++i) {
                _statVectors(K, i) = std::exp(_thermStateEnergies(K) - _biasedConfEnergies(K, i));
            }
        }
        return statVectors;
    }

    void computeMarkovStateEnergies() {
        auto _markovStateEnergies = markovStateEnergies.template mutable_unchecked<1>();

        // first reset all confirmation energies to infinity
        for (int i = 0; i < nMarkovStates; i++) {
            _markovStateEnergies(i) = inf;
        }

        for (int K = 0; K < nThermStates; ++K) {
            auto _dtraj_K = input->dtraj(K);
            auto _biasMatrix_K = input->biasMatrix(K);
            int trajLength = input->sequenceLength(K);

            computeMarkovStateEnergiesForSingleTrajectory(_biasMatrix_K, _dtraj_K, trajLength);
        }
    }

    template<typename BiasMatrixK, typename Dtraj>
    void
    computeMarkovStateEnergiesForSingleTrajectory(const BiasMatrixK &_biasMatrix_K,
                                                  const Dtraj &_dtraj,
                                                  int trajlength /*todo this is just dtraj length*/) {
        auto _modifiedStateCountsLog = modifiedStateCountsLog.template unchecked<2>();
        auto _markovStateEnergies = markovStateEnergies.template mutable_unchecked<1>();

        int i, K, x, o;
        dtype divisor;
        /* assume that markovStateEnergies was set to INF by the caller on the first call */
        for (x = 0; x < trajlength; ++x) {
            i = _dtraj(x);
            if (i < 0) continue;
            o = 0;
            for (K = 0; K < nThermStates; ++K) {
                if (-inf == _modifiedStateCountsLog(K, i)) continue;
                scratchT[o++] =
                        _modifiedStateCountsLog(K, i) - _biasMatrix_K(x, K);
            }
            divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratchT.begin(), o);
            _markovStateEnergies(i) = -numeric::kahan::logsumexp_pair(-_markovStateEnergies(i), -divisor);
        }
    }

    void computeThermStateEnergies() {
        auto _biasedConfEnergies = biasedConfEnergies.template unchecked<2>();
        auto _thermStateEnergies = thermStateEnergies.template mutable_unchecked<1>();

        for (int K = 0; K < nThermStates; ++K) {
            for (int i = 0; i < nMarkovStates; ++i)
                scratchM[i] = -_biasedConfEnergies(K, i);
            _thermStateEnergies(K) = -numeric::kahan::logsumexp_sort_kahan_inplace(scratchM.begin(), nMarkovStates);
        }
    }

    void shiftEnergiesToHaveZeroMinimum() {
        auto _biasedConfEnergies = biasedConfEnergies.template mutable_unchecked<2>();
        auto _thermStateEnergies = thermStateEnergies.template mutable_unchecked<1>();

        dtype shift = 0;

        for (int K = 0; K < nThermStates; ++K) {
            for (int i = 0; i < nMarkovStates; ++i) {
                if (_biasedConfEnergies(K, i) < shift) {
                    shift = _biasedConfEnergies(K, i);
                }
            }
        }
        for (int K = 0; K < nThermStates; ++K) {
            _thermStateEnergies(K) -= shift;

            for (int i = 0; i < nMarkovStates; ++i) {
                _biasedConfEnergies(K, i) -= shift;
            }
        }
    }

    void normalize() {
        auto _biasedConfEnergies = biasedConfEnergies.template mutable_unchecked<2>();
        auto _markovStateEnergies = markovStateEnergies.template mutable_unchecked<1>();
        auto _thermStateEnergies = thermStateEnergies.template mutable_unchecked<1>();

        for (int i = 0; i < nMarkovStates; ++i) {
            scratchM[i] = -_markovStateEnergies(i);
        }
        dtype f0 = -numeric::kahan::logsumexp_sort_kahan_inplace(scratchM.begin(), nMarkovStates);

        for (int i = 0; i < nMarkovStates; ++i) {
            _markovStateEnergies(i) -= f0;
            _thermStateEnergies(i) -= f0;
            for (int K = 0; K < nThermStates; ++K) {
                _biasedConfEnergies(K, i) -= f0;
            }
        }
    }

    np_array_nfc<dtype> estimateTransitionMatrices() {
        auto _biasedConfEnergies = biasedConfEnergies.template unchecked<2>();
        auto _lagrangianMultLog = lagrangianMultLog.template unchecked<2>();
        auto _modifiedStateCountsLog = modifiedStateCountsLog.template unchecked<2>();

        auto _transitionCounts = input->transitionCounts();
        auto _transitionMatrices = transitionMatrices.template mutable_unchecked<3>();

        int C;
        dtype divisor, maxSum;
        for (int K = 0; K < nThermStates; ++K) {
            for (int i = 0; i < nMarkovStates; ++i) {
                scratchM[i] = 0.0;
                for (int j = 0; j < nMarkovStates; ++j) {
                    _transitionMatrices(K, i, j) = 0.0;
                    C = _transitionMatrices(K, i, j) + _transitionCounts(K, j, i);
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
            for (int i = 0; i < nMarkovStates; ++i) if (scratchM[i] > maxSum) maxSum = scratchM[i];
            if (maxSum == 0) maxSum = 1.0; /* completely empty T matrix -> generate Id matrix */
            for (int i = 0; i < nMarkovStates; ++i) {
                for (int j = 0; j < nMarkovStates; ++j) {
                    if (i == j) {
                        _transitionMatrices(K, i, i) =
                                (_transitionMatrices(K, i, i) + maxSum - scratchM[i]) / maxSum;
                        if (0 == _transitionMatrices(K, i, i) && 0 < _transitionCounts(K, i, i))
                            fprintf(stderr, "# Warning: zero diagonal element T[%d,%d] with non-zero counts.\n", i,
                                    i);
                    } else {
                        _transitionMatrices(K, i, j) = _transitionMatrices(K, i, j) / maxSum;
                    }
                }
            }
        }
        return transitionMatrices;
    }

    /* TRAM log-likelihood that comes from the terms containing discrete quantities */
    dtype discreteLogLikelihoodLowerBound() {
        auto _biasedConfEnergies = biasedConfEnergies.template unchecked<2>();
        auto _transitionCounts = input->transitionCounts();
        auto _stateCounts = input->stateCounts();
        auto _transitionMatrices = transitionMatrices.template unchecked<3>();

        int CKij;

        /* \sum_{i,j,k}c_{ij}^{(k)}\log p_{ij}^{(k)} */
        dtype a = 0;
        estimateTransitionMatrices();

        for (int K = 0; K < nThermStates; ++K) {
            for (int i = 0; i < nMarkovStates; ++i) {
                for (int j = 0; j < nMarkovStates; ++j) {
                    CKij = _transitionCounts(K, i, j);
                    if (0 == CKij) continue;
                    if (i == j) {
                        a += ((dtype) CKij + THERMOTOOLS_TRAM_PRIOR) * log(_transitionMatrices(K, i, j));
                    } else {
                        a += CKij * log(_transitionMatrices(K, i, j));
                    }
                }
            }
        }
        /* \sum_{i,k}N_{i}^{(k)}f_{i}^{(k)} */
        dtype b = 0;
        for (int K = 0; K < nThermStates; ++K) {
            for (int i = 0; i < nMarkovStates; ++i) {
                if (_stateCounts(K, i) > 0)
                    b += (_stateCounts(K, i) + THERMOTOOLS_TRAM_PRIOR) * _biasedConfEnergies(K, i);
            }
        }
        return a + b;
    }

    // TODO: fix this
//    template<typename dtype>
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
_bar_df(np_array_nfc<dtype> db_IJ, int L1, np_array_nfc<dtype> db_JI, int L2, np_array_nfc<dtype> scratch) {
    py::buffer_info db_IJ_buf = db_IJ.request();
    py::buffer_info db_JI_buf = db_JI.request();
    py::buffer_info scratch_buf = scratch.request();

    dtype *db_IJ_ptr = (dtype *) db_IJ_buf.ptr;
    dtype *db_JI_ptr = (dtype *) db_JI_buf.ptr;
    dtype *scratch_ptr = (dtype *) scratch_buf.ptr;

    int i;
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
}
