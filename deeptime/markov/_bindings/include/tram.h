//
// Created by Maaike on 15/11/2021.
//

#pragma once

#include <cstdio>
#include <cassert>
#include "common.h"
#include "kahan_summation.h"

namespace deeptime {
namespace tram {

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

template<typename dtype>
class TRAMInput {
public:
    TRAMInput(const np_array_nfc<int> &state_counts, const np_array_nfc<int> transition_counts,
              py::list bias_matrix, py::list dtrajs)
            : state_counts(state_counts),
              transition_counts(transition_counts),
              bias_matrix(bias_matrix),
              dtrajs(dtrajs) {
        validateInput();
    }

    TRAMInput() {}

    void validateInput() {
        throwIfInvalid(dtrajs.size() != bias_matrix.size(),
                       "Input invalid. Number of trajectories should be equal to the size of the first dimension of the bias matrix.");

        for(int i = 0; i < dtrajs.size(); ++i) {
            auto dtrajs_i =  py::cast<np_array_nfc<int>>(dtrajs[i]);
            auto bias_matrix_i = py::cast<np_array_nfc<dtype>>(bias_matrix[i]);
            auto dtrajs_buf_i = dtrajs_i.request();
            auto bias_matrix_buf_i = bias_matrix_i.request();

            throwIfInvalid(dtrajs_buf_i.ndim == 1, "dtraj at index {i} has an incorrect number of dimension. ndims should be 1.");
            throwIfInvalid(bias_matrix_buf_i.ndim == 2, "get_bias_matrix_row at index {i} has an incorrect number of dimension. ndims should be 2.");
            throwIfInvalid(dtrajs_buf_i.shape[0] == bias_matrix_buf_i.shape[0], "dtraj and get_bias_matrix_row at index {i} should be of equal length.");
            throwIfInvalid(bias_matrix_buf_i.shape[1] == transition_counts.shape()[0], "get_bias_matrix_row{i}.shape[1] should be equal to get_transition_counts.shape[0].");

            throwIfInvalid(get_dtraj(i).flags().c_contiguous, "dtraj at index {i} is not contiguous.");
            throwIfInvalid(get_bias_matrix_row(i).flags().c_contiguous, "_bias_matrix at index {i} is not contiguous.");
        }
    }

    void throwIfInvalid(bool isValid, std::string message) {
        if (!isValid) {
            throw std::invalid_argument(message);
        }
    }

    auto get_bias_matrix_row(int K) {
        np_array_nfc<dtype> bias_matrix_K =  py::cast<np_array_nfc<dtype>>(bias_matrix[K]);
        return bias_matrix_K.template unchecked<2>();
    }

    auto get_dtraj(int K) {
        np_array_nfc<int> dtrajs_K =  py::cast<np_array_nfc<int>>(dtrajs[K]);
        return dtrajs_K.template unchecked<1>();
    }

    auto get_transition_counts() {
        return transition_counts.template unchecked<3>();
    }

    auto get_state_counts() {
        return state_counts.template unchecked<2>();
    }

private:
    np_array<int> state_counts;
    np_array<int> transition_counts;
    py::list dtrajs;
    py::list bias_matrix;
};

template<typename dtype>
struct TRAM {
// # TODO: wrap vector in something nice for happy indexing
// TODO: make this a vector (no need for np_array)
    np_array<dtype> biased_conf_energies;
    np_array<dtype> log_lagrangian_mult;
    np_array<dtype> modified_state_counts_log;

//    std::vector<dtype> biased_conf_energies;
//    std::vector<dtype> log_lagrangian_mult;
//    std::vector<dtype> modified_state_counts_log;


    np_array<dtype> conf_energies;
    np_array<dtype> therm_energies;
    np_array<dtype> transition_matrices;

    TRAMInput<dtype> input;

    int n_therm_states;
    int n_markov_states;

    int save_convergence_info;

    // scratch matrices used to facilitate calculation of logsumexp
    np_array<dtype> scratch_M;
    np_array<dtype> scratch_T;

    TRAM(const np_array_nfc<int> state_counts, np_array_nfc<int> transition_counts, py::list dtrajs,
         py::list bias_matrix, int save_convergence_info = 0)
            : n_therm_states(state_counts.shape()[0]),
              n_markov_states(state_counts.shape()[1]),
              save_convergence_info(save_convergence_info) {

        input = TRAMInput<dtype>(state_counts, transition_counts, dtrajs, bias_matrix);


        biased_conf_energies = np_array<dtype>({n_therm_states, n_markov_states});
        log_lagrangian_mult = np_array<dtype>({n_therm_states, n_markov_states});
        modified_state_counts_log = np_array<dtype>({n_therm_states, n_markov_states});

        transition_matrices = np_array<dtype>({n_therm_states, n_markov_states, n_markov_states});

        conf_energies = np_array<dtype>({n_markov_states});
        therm_energies = np_array<dtype>({n_therm_states});

        scratch_M = np_array<dtype>({n_markov_states});
        scratch_T = np_array<dtype>({n_therm_states});

        initLagrangianMult();



//        old_biased_conf_energies = self.biased_conf_energies.copy()
//        old_log_lagrangian_mult = self.log_lagrangian_mult.copy()
//        old_stat_vectors = np.zeros(shape=get_state_counts.shape, dtype=np.float64)
//        old_therm_energies = np.zeros(shape=get_transition_counts.shape[0], dtype=np.float64)

    }


    void estimate(int maxiter = 1000) {

        int iterationCount = 0;

//#TODO: do something with these. logging?
        increments = []
        log_likelihoods = []


        for (int m = 0; m < maxiter; ++m) {


            iterationCount += 1

            updateLagrangianMult()
            updateStateCounts();
            updateBiasedConfEnergies()

            self.therm_energies = np.zeros(shape = self.n_therm_states, dtype = np.float64)
            tram.get_therm_energies(self.biased_conf_energies, self.n_therm_states, self.n_markov_states,
                                    scratch_M, self.therm_energies)
            stat_vectors = np.exp(self.therm_energies[:, np.newaxis] -self.biased_conf_energies)
            delta_therm_energies = np.abs(self.therm_energies - old_therm_energies)
            delta_stat_vectors = np.abs(stat_vectors - old_stat_vectors)
            err = max(np.max(delta_therm_energies), np.max(delta_stat_vectors))
            if iteration_count == self.save_convergence_info:
            iteration_count = 0
            increments.append(err)
            log_likelihoods.append(l)

            if err < self.maxerr:
            break
            else:
            shift = np.min(self.biased_conf_energies)
            self.biased_conf_energies -= shift
            old_biased_conf_energies[:] = self.biased_conf_energies
            old_log_lagrangian_mult[:] = self.log_lagrangian_mult[:]
            old_therm_energies[:] = self.therm_energies[:] -shift
            old_stat_vectors[:] = stat_vectors[:]

            self.markov_energies = self.get_conf_energies(bias_matrix, markov_state_sequences, log_R_K_i, scratch_T)
            tram.get_therm_energies(self.biased_conf_energies, self.n_therm_states, self.n_markov_states, scratch_M,
                                    self.therm_energies)
            tram.normalize(self.markov_energies, self.biased_conf_energies, self.therm_energies, self.n_therm_states,
                           self.n_markov_states, scratch_M)
            if err >= self.maxerr:
            import warnings
            warnings.warn(f
            "TRAM did not converge: last increment = {err}", UserWarning)
        }
    }

    void initLagrangianMult() {
        auto _transition_counts = input.get_transition_counts();
        auto _log_lagrangian_mult = log_lagrangian_mult.template mutable_unchecked<2>();

        for (int K = 0; K < n_therm_states; ++K) {
            for (int i = 0; i < n_markov_states; ++i) {
                dtype sum = 0.0;
                for (int j = 0; j < n_markov_states; ++j) {
                    sum += (_transition_counts(K, i, j) +
                            _transition_counts(K, j, i));
                }
                _log_lagrangian_mult(K, i) = std::log(sum / 2);
            }
        }
    }

    // TODO: Make this just return whatever is needed in stead of this weird method
    np_array<dtype> updateLagrangianMult() {
        auto _log_lagrangian_mult = log_lagrangian_mult.template unchecked<2>();
        auto _biased_conf_energies = biased_conf_energies.template unchecked<2>();

        auto _transition_counts = input.get_transition_counts();
        auto _state_counts = input.get_state_counts();

        auto _scratch_M = scratch_M.template mutable_unchecked<1>();

        auto new_log_lagrangian_mult = np_array<dtype>({n_therm_states, n_markov_states});
        auto _new_log_lagrangian_mult = new_log_lagrangian_mult.template mutable_unchecked<2>();

        int CK, CKij;
        dtype divisor;

        for (int K = 0; K < n_therm_states; ++K) {
            for (int i = 0; i < n_markov_states; ++i) {
                if (0 == _state_counts(K, i)) {
                    _new_log_lagrangian_mult(K, i) = -INFINITY;
                    continue;
                }
                int o = 0;
                for (int j = 0; j < n_markov_states; ++j) {
                    CKij = _transition_counts(K, i, j);
                    /* special case: most variables cancel out, here */
                    if (i == j) {
                        _scratch_M(o++) = (0 == CKij) ?
                                          THERMOTOOLS_TRAM_LOG_PRIOR : log(THERMOTOOLS_TRAM_PRIOR + (dtype) CKij);
                        continue;
                    }
                    CK = CKij + _transition_counts(K, j, i);
                    /* special case */
                    if (0 == CK) continue;
                    /* regular case */
                    divisor = numeric::kahan::logsumexp_pair(
                            _log_lagrangian_mult(K, j) - _biased_conf_energies(K, i)
                            - _log_lagrangian_mult(K, i) + _biased_conf_energies(K, j), 0.0);
                    _scratch_M(o++) = log((dtype) CK) - divisor;
                }
                _new_log_lagrangian_mult(K, i) = numeric::kahan::logsumexp_sort_kahan_inplace(scratch_M, o);
            }
        }
        return new_log_lagrangian_mult;
    }

    np_array<dtype> updateBiasedConfEnergies(int return_log_l = 0) {
        dtype log_L = 0.0;
//        new_biased_conf_energies[:] = _np.inf

        auto new_biased_conf_energies = np_array<dtype>({n_therm_states, n_markov_states});

        for (int K = 0; K < n_therm_states; ++K){
            log_L += update_biased_conf_energies(K, new_biased_conf_energies, return_log_l);
        }
        return new_biased_conf_energies;
    }

    np_array<dtype>
    update_biased_conf_energies(int therm_state, np_array<dtype> new_biased_conf_energies, int return_log_L) {
        auto _new_biased_conf_energies = new_biased_conf_energies.template mutable_unchecked<2>();

        auto _modified_state_counts_log = modified_state_counts_log.template unchecked<2>();

        auto _dtraj = input.get_dtraj(therm_state);
        auto _bias_matrix = input.get_bias_matrix_row(therm_state);

        auto _scratch_T = scratch_T.template mutable_unchecked<1>();

        int seq_length = _dtraj.shape[0];

        dtype divisor, log_L;

        /* assume that new_biased_conf_energies have been set to INF by the caller in the first call */
        for (int x = 0; x < seq_length; ++x) {
            int i = _dtraj(x);
            if (i < 0) continue; /* skip frames that have negative Markov state indices */
            int o = 0;
            for (int K = 0; K < n_therm_states; ++K) {

                /* applying Hao's speed-up recomendation */
                if (-INFINITY == _modified_state_counts_log(K, i)) continue;
                _scratch_T(o++) = _modified_state_counts_log(K, i) - _bias_matrix(x, K);
            }
            divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratch_T, o);

            for (int K = 0; K < n_therm_states; ++K) {
                _new_biased_conf_energies(K, i) = -numeric::kahan::logsumexp_pair(
                        -_new_biased_conf_energies(K, i), //TODO: THIS SHOULD BE INF?????
                        -(divisor + _bias_matrix(x, K)));
            }
        }

        // TODO: mechanism to save this progress indicator. Maybe a callback?
        if (return_log_L) {
            /* -\sum_{x}\log\sum_{l}R_{i(x)}^{(l)}e^{-b^{(l)}(x)+f_{i(x)}^{(l)}} */
            log_L = 0;
            for (int x = 0; x < seq_length; ++x) {
                int o = 0;
                int i = _dtraj(x);
                if (i < 0) continue;
                for (int K = 0; K < n_therm_states; ++K) {
                    if (_modified_state_counts_log(K, i) > 0)
                        _scratch_T(o++) =
                                _modified_state_counts_log(K, i) - _bias_matrix(x, K);
                }
                log_L -= numeric::kahan::logsumexp_sort_kahan_inplace(scratch_T, o);
            }
        }
        return log_L;
    }

//    template<typename dtype, bool trammbar = false>
    void updateStateCounts() {
        auto _biased_conf_energies = biased_conf_energies.template unchecked<2>();
        auto _log_lagrangian_mult = log_lagrangian_mult.template unchecked<2>();
        auto _modified_state_counts_log = modified_state_counts_log.template unchecked<2>();

        auto _state_counts = input.get_state_counts();
        auto _transition_counts = input.get_transition_counts();

        auto _scratch_M = scratch_M.template mutable_unchecked<1>();

        auto new_biased_conf_energies = np_array<dtype>({n_therm_states, n_markov_states});
        auto _new_biased_conf_energies = new_biased_conf_energies.template mutable_unchecked<2>();


        int o;
        int Ci, CK, CKij, CKji, NC;
        dtype divisor, R_addon;

        for (int K = 0; K < n_therm_states; ++K) {
            for (int i = 0; i < n_markov_states; ++i) {
                if (0 == _state_counts(K, i)) /* applying Hao's speed-up recomendation */
                {
                    _modified_state_counts_log(K, i) = -INFINITY;
                    continue;
                }
                Ci = 0;
                o = 0;
                for (int j = 0; j < n_markov_states; ++j) {
                    CKij = _transition_counts(K, i, j);
                    CKji = _transition_counts(K, j, i);
                    Ci += CKji;
                    /* special case: most variables cancel out, here */
                    if (i == j) {
                        _scratch_M(o) = (0 == CKij) ? THERMOTOOLS_TRAM_LOG_PRIOR : log(
                                THERMOTOOLS_TRAM_PRIOR + (dtype) CKij);
                        _scratch_M(o++) += _biased_conf_energies(K, i);
                        continue;
                    }
                    CK = CKij + CKji;
                    /* special case */
                    if (0 == CK) continue;
                    /* regular case */
                    divisor = numeric::kahan::logsumexp_pair(
                            _log_lagrangian_mult(K, j) - _biased_conf_energies(K, i),
                            _log_lagrangian_mult(K, i) - _biased_conf_energies(K, j));
                    _scratch_M(o++) = log((dtype) CK) + _log_lagrangian_mult(K, j) - divisor;
                }
                NC = _state_counts(K, i) - Ci;
                R_addon = (0 < NC) ? log((dtype) NC) + _biased_conf_energies(K, i) : -INFINITY; /* IGNORE PRIOR */
                _modified_state_counts_log(K, i) = numeric::kahan::logsumexp_pair(
                        numeric::kahan::logsumexp_sort_kahan_inplace(scratch_M, o), R_addon);
            }
        }

//        if constexpr(trammbar) {
        // todo
        /*if(equilibrium_therm_state_counts && therm_energies)
        {
            for(K=0; K<n_therm_states; ++K)
            {
                KM = K * n_conf_states;
                for(i=0; i<n_conf_states; ++i)
                    log_R_K_i[KM + i] += log(overcounting_factor);
            }
            for(K=0; K<n_therm_states; ++K)
            {
                if(0 < equilibrium_therm_state_counts[K])
                {
                    KM = K * n_conf_states;
                    for(i=0; i<n_conf_states; ++i)
                    {
                        Ki = KM + i;
                        log_R_K_i[Ki] = logsumexp_pair(log_R_K_i[Ki], log(equilibrium_therm_state_counts[K]) + therm_energies[K]);
                    }
                }
            }
        } */
//        }

    }

    void compute_conf_energies(np_array<dtype> bias_energy_sequence,
                               np_array<int> dtraj, int seq_length) {
        auto _dtraj = dtraj.template unchecked<1>();
        auto _bias_energy_sequence = bias_energy_sequence.template unchecked<2>();

        auto _modified_state_counts_log = modified_state_counts_log.template unchecked<2>();

        auto _scratch_T = scratch_T.template mutable_unchecked<1>();

        auto _conf_energies = conf_energies.template mutable_checked<1>();

        int i, K, x, o;
        dtype divisor;
        /* assume that conf_energies was set to INF by the caller on the first call */
        for (x = 0; x < seq_length; ++x) {
            i = _dtraj(x);
            if (i < 0) continue;
            o = 0;
            for (K = 0; K < n_therm_states; ++K) {
                if (-INFINITY == _modified_state_counts_log(K, i)) continue;
                _scratch_T(o++) =
                        _modified_state_counts_log(K, i) - _bias_energy_sequence(x, K);
            }
            divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratch_T, o);
            _conf_energies(i) = -numeric::kahan::logsumexp_pair(-_conf_energies(i), -divisor);
        }
    }

    void compute_therm_energies() {
        auto _biased_conf_energies = biased_conf_energies.template unchecked<2>();
        auto _therm_energies = therm_energies.template mutable_unchecked<1>();
        auto _scratch_M = scratch_M.template mutable_unchecked<1>();

        for (int K = 0; K < n_therm_states; ++K) {
            for (int i = 0; i < n_markov_states; ++i)
                _scratch_M(i) = -_biased_conf_energies(K, i);
            _therm_energies(K) = -numeric::kahan::logsumexp_sort_kahan_inplace(scratch_M, n_markov_states);
        }
    }

    void normalize() {
        auto _biased_conf_energies = biased_conf_energies.template unchecked<2>();
        auto _conf_energies = therm_energies.template mutable_unchecked<1>();
        auto _therm_energies = therm_energies.template mutable_unchecked<1>();
        auto _scratch_M = scratch_M.template mutable_unchecked<1>();

        for (int i = 0; i < n_markov_states; ++i) {
            _scratch_M(i) = -_conf_energies(i);
        }
        auto f0 = -numeric::kahan::logsumexp_sort_kahan_inplace(scratch_M, n_markov_states);

        for (int i = 0; i < n_markov_states; ++i) {
            _conf_energies(i) -= f0;
            _therm_energies(i) -= f0;
            for (int K = 0; K < n_therm_states; ++K) {
                _biased_conf_energies(i) -= f0;
            }
        }
    }

    void estimate_transition_matrices() {
        auto _biased_conf_energies = biased_conf_energies.template unchecked<2>();
        auto _log_lagrangian_mult = log_lagrangian_mult.template unchecked<2>();
        auto _modified_state_counts_log = modified_state_counts_log.template unchecked<2>();

        auto _transition_counts = input.get_transition_counts();
        auto sum = scratch_M.template mutable_unchecked<1>();

        auto _transition_matrices = transition_matrices.template mutable_unchecked<33>();

        int C;
        dtype divisor, max_sum;
        for (int K = 0; K < n_therm_states; ++K) {
            for (int i = 0; i < n_markov_states; ++i) {
                sum(i) = 0.0;
                for (int j = 0; j < n_markov_states; ++j) {
                    _transition_matrices(i, j) = 0.0;
                    C = _transition_matrices(i, j) + _transition_counts(K, j, i);
                    /* special case: this element is zero */
                    if (0 == C) continue;
                    if (i == j) {
                        /* special case: diagonal element */
                        _transition_matrices(K, i, j) = 0.5 * C * exp(-_log_lagrangian_mult(K, i));
                    } else {
                        /* regular case */
                        divisor = numeric::kahan::logsumexp_pair(
                                _log_lagrangian_mult(K, j) - _biased_conf_energies(K, i),
                                _log_lagrangian_mult(K, i) - _biased_conf_energies(K, j));
                        _transition_matrices(K, i, j) = C * exp(-(_biased_conf_energies(K, j) + divisor));
                    }
                    sum(i) += _transition_matrices(K, i, j);
                }
            }
            /* normalize T matrix */ /* TODO: unify with util._renormalize_transition_matrix? */
            max_sum = 0;
            for (int i = 0; i < n_markov_states; ++i) if (sum(i) > max_sum) max_sum = sum(i);
            if (max_sum == 0) max_sum = 1.0; /* completely empty T matrix -> generate Id matrix */
            for (int i = 0; i < n_markov_states; ++i) {
                for (int j = 0; j < n_markov_states; ++j) {
                    if (i == j) {
                        _transition_matrices(K, i, i) = (_transition_matrices(K, i, i) + max_sum - sum(i)) / max_sum;
                        if (0 == _transition_matrices(K, i, i) && 0 < _transition_counts(K, i, i))
                            fprintf(stderr, "# Warning: zero diagonal element T[%d,%d] with non-zero counts.\n", i, i);
                    } else {
                        _transition_matrices(K, i, j) = _transition_matrices(K, i, j) / max_sum;
                    }
                }
            }
        }
    }

    /* TRAM log-likelihood that comes from the terms containing discrete quantities */
    dtype discrete_log_likelihood_lower_bound() {
        auto _biased_conf_energies = biased_conf_energies.template unchecked<2>();
        auto _transition_counts = input.get_transition_counts();
        auto _state_counts = input.get_state_counts();
        auto _transition_matrices = transition_matrices.template unchecked<3>();

        int CKij;

        /* \sum_{i,j,k}c_{ij}^{(k)}\log p_{ij}^{(k)} */
        dtype a = 0;
        estimate_transition_matrices();
        for (int K = 0; K < n_therm_states; ++K) {
            for (int i = 0; i < n_markov_states; ++i) {
                for (int j = 0; j < n_markov_states; ++j) {
                    CKij = _transition_counts(K, i, j);
                    if (0 == CKij) continue;
                    if (i == j) {
                        a += ((dtype) CKij + THERMOTOOLS_TRAM_PRIOR) * log(_transition_matrices(K, i, j));
                    } else {
                        a += CKij * log(_transition_matrices(K, i, j));
                    }
                }
            }
        }
        /* \sum_{i,k}N_{i}^{(k)}f_{i}^{(k)} */
        dtype b = 0;
        for (int K = 0; K < n_therm_states; ++K) {
            for (int i = 0; i < n_markov_states; ++i) {
                if (_state_counts(K, i) > 0)
                    b += (_state_counts(K, i) + THERMOTOOLS_TRAM_PRIOR) * _biased_conf_energies(K, i);
            }
        }
        return a + b;
    }

    // TODO: fix this
//    template<typename dtype>
//    void get_pointwise_unbiased_free_energies(int K) {
//        auto _dtrajs = input.get_dtraj();
//        auto _bias_matrix = input.get_bias_matrix_row();
//
//        auto _therm_energies = therm_energies.template unchecked<1>();
//        auto _modified_state_counts_log = modified_state_counts_log.template unchecked<2>();
//
//        auto _scratch_T = scratch_T.template unchecked<1>();
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
//                _pointwise_unbiased_free_energies(x) = INFINITY;
//                continue;
//            }
//            o = 0;
//            for (L = 0; L < n_therm_states; ++L) {
//                if (-INFINITY == _modified_state_counts_log(L, i)) continue;
//                _scratch_T(o++) =
//                        _modified_state_counts_log(L, i) - _bias_matrix(x, L);
//            }
//            log_divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratch_T, o);
//            if (K == -1)
//                pointwise_unbiased_free_energies_ptr[x] = log_divisor;
//            else
//                pointwise_unbiased_free_energies_ptr[x] =
//                        bias_energy_sequence_ptr[x * n_therm_states + k] + log_divisor - therm_energies_ptr[k];
//        }
//    }
//};
};

template<typename dtype>
extern dtype _bar_df(np_array<dtype> db_IJ, int L1, np_array<dtype> db_JI, int L2, np_array<dtype> scratch) {
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
    ln_avg1 = numeric::kahan::logsumexp_sort_kahan_inplace(scratch_ptr, L1);
    for (i = 0; i < L1; i++) {
        scratch_ptr[i] = db_JI_ptr[i] > 0 ? 0 : db_JI_ptr[i];
    }
    ln_avg2 = numeric::kahan::logsumexp_sort_kahan_inplace(scratch_ptr, L2);
    return ln_avg2 - ln_avg1;
}
}
}
