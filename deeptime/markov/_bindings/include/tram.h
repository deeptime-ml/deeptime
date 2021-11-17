//
// Created by Maaike on 15/11/2021.
//

#pragma once

#include <stdio.h>
#include <assert.h>
#include "common.h"
#include "logsumexp.h"
//
//template<typename dtype>
//int tram_test(dtype testVar) {
//    return testVar + 1;
//}

double THERMOTOOLS_TRAM_PRIOR = 0.0;
double THERMOTOOLS_TRAM_LOG_PRIOR = 1.0;

template<typename dtype>
void init_lagrangian_mult(np_array<dtype> count_matrices, int n_therm_states, int n_conf_states,
                          np_array<dtype> log_lagrangian_mult) {
    int i, j, K;
    int MM = n_conf_states * n_conf_states, KMM;
    dtype sum;

    py::buffer_info count_matrices_buf = count_matrices.request();
    dtype *count_matrices_ptr = (dtype *) count_matrices_buf.ptr;

    py::buffer_info log_lagrangian_mult_buf = log_lagrangian_mult.request();
    dtype *log_lagrangian_mult_ptr = (dtype *) log_lagrangian_mult_buf.ptr;

    for (K = 0; K < n_therm_states; ++K) {
        KMM = K * MM;
        for (i = 0; i < n_conf_states; ++i) {
            sum = 0.0;
            for (j = 0; j < n_conf_states; ++j) {
                sum += 0.5 * (count_matrices_ptr[KMM + i * n_conf_states + j] +
                              count_matrices_ptr[KMM + j * n_conf_states + i]);
            }
            log_lagrangian_mult_ptr[K * n_conf_states + i] = (dtype) log(sum);
        }
    }
}

// TODO: Make this just return whatever is needed in stead of this weird method
template<typename dtype>
void update_lagrangian_mult(
        np_array<dtype> log_lagrangian_mult, np_array<dtype> biased_conf_energies, np_array<int> count_matrices,
        np_array<int> state_counts, int n_therm_states, int n_conf_states, np_array<dtype> scratch_M,
        np_array<dtype> new_log_lagrangian_mult) {
    // TODO: define get_pointer_from_array() function for this shit
    py::buffer_info log_lagrangian_mult_buf = log_lagrangian_mult.request();
    py::buffer_info biased_conf_energies_buf = biased_conf_energies.request();
    py::buffer_info count_matrices_buf = count_matrices.request();
    py::buffer_info state_counts_buf = state_counts.request();
    py::buffer_info scratch_M_buf = scratch_M.request();
    py::buffer_info new_log_lagrangian_mult_buf = new_log_lagrangian_mult.request();

    dtype *log_lagrangian_mult_ptr = (dtype *) log_lagrangian_mult_buf.ptr;
    dtype *biased_conf_energies_ptr = (dtype *) biased_conf_energies_buf.ptr;
    int *count_matrices_ptr = (int *) count_matrices_buf.ptr;
    int *state_counts_ptr = (int *) state_counts_buf.ptr;
    dtype *scratch_M_ptr = (dtype *) scratch_M_buf.ptr;
    dtype *new_log_lagrangian_mult_ptr = (dtype *) new_log_lagrangian_mult_buf.ptr;

    int i, j, K, o;
    int Ki, Kj, KM, KMM;
    int CK, CKij;
    double divisor;
    for (K = 0; K < n_therm_states; ++K) {
        KM = K * n_conf_states;
        KMM = KM * n_conf_states;
        for (i = 0; i < n_conf_states; ++i) {
            Ki = KM + i;
            if (0 == state_counts_ptr[Ki]) {
                new_log_lagrangian_mult_ptr[Ki] = -INFINITY;
                continue;
            }
            o = 0;
            for (j = 0; j < n_conf_states; ++j) {
                CKij = count_matrices_ptr[KMM + i * n_conf_states + j];
                /* special case: most variables cancel out, here */
                if (i == j) {
                    scratch_M_ptr[o++] = (0 == CKij) ?
                                         THERMOTOOLS_TRAM_LOG_PRIOR : log(THERMOTOOLS_TRAM_PRIOR + (dtype) CKij);
                    continue;
                }
                CK = CKij + count_matrices_ptr[KMM + j * n_conf_states + i];
                /* special case */
                if (0 == CK) continue;
                /* regular case */
                Kj = KM + j;
                divisor = _logsumexp_pair(
                        log_lagrangian_mult_ptr[Kj] - biased_conf_energies_ptr[Ki] - log_lagrangian_mult_ptr[Ki] +
                        biased_conf_energies_ptr[Kj], 0.0);
                scratch_M_ptr[o++] = log((dtype) CK) - divisor;
            }
            new_log_lagrangian_mult_ptr[Ki] = _logsumexp_sort_kahan_inplace(scratch_M_ptr, o);
        }
    }
}

template<typename dtype>
dtype update_biased_conf_energies(
        np_array<dtype> bias_energy_sequence, np_array<int> state_sequence, int seq_length, np_array<dtype> log_R_K_i,
        int n_therm_states, int n_conf_states, np_array<dtype> scratch_T, np_array<dtype> new_biased_conf_energies,
        int return_log_L) {
    py::buffer_info bias_energy_sequence_buf = bias_energy_sequence.request();
    py::buffer_info state_sequence_buf = state_sequence.request();
    py::buffer_info log_R_K_i_buf = log_R_K_i.request();
    py::buffer_info scratch_T_buf = scratch_T.request();
    py::buffer_info new_biased_conf_energies_buf = new_biased_conf_energies.request();

    dtype *bias_energy_sequence_ptr = (dtype *) bias_energy_sequence_buf.ptr;
    int *state_sequence_ptr = (int *) state_sequence_buf.ptr;
    dtype *log_R_K_i_ptr = (dtype *) log_R_K_i_buf.ptr;
    dtype *scratch_T_ptr = (dtype *) scratch_T_buf.ptr;
    dtype *new_biased_conf_energies_ptr = (dtype *) new_biased_conf_energies_buf.ptr;

    int i, K, x, o, Ki;
    int KM;
    dtype divisor, log_L;

    /* assume that new_biased_conf_energies have been set to INF by the caller in the first call */
    for (x = 0; x < seq_length; ++x) {
        i = state_sequence_ptr[x];
        if (i < 0) continue; /* skip frames that have negative Markov state indices */
        o = 0;
        for (K = 0; K < n_therm_states; ++K) {
            assert(K < n_therm_states);
            assert(K >= 0);
            /* applying Hao's speed-up recomendation */
            if (-INFINITY == log_R_K_i_ptr[K * n_conf_states + i]) continue;
            scratch_T_ptr[o++] =
                    log_R_K_i_ptr[K * n_conf_states + i] - bias_energy_sequence_ptr[x * n_therm_states + K];
        }
        divisor = _logsumexp_sort_kahan_inplace(scratch_T_ptr, o);

        for (K = 0; K < n_therm_states; ++K) {
            new_biased_conf_energies_ptr[K * n_conf_states + i] = -_logsumexp_pair(
                    -new_biased_conf_energies_ptr[K * n_conf_states + i],
                    -(divisor + bias_energy_sequence_ptr[x * n_therm_states + K]));
        }
    }

    if (return_log_L) {
        /* -\sum_{x}\log\sum_{l}R_{i(x)}^{(l)}e^{-b^{(l)}(x)+f_{i(x)}^{(l)}} */
        log_L = 0;
        for (x = 0; x < seq_length; ++x) {
            o = 0;
            i = state_sequence_ptr[x];
            if (i < 0) continue;
            for (K = 0; K < n_therm_states; ++K) {
                KM = K * n_conf_states;
                Ki = KM + i;
                if (log_R_K_i_ptr[Ki] > 0)
                    scratch_T_ptr[o++] =
                            log_R_K_i_ptr[Ki] - bias_energy_sequence_ptr[x * n_therm_states + K];
            }
            log_L -= _logsumexp_sort_kahan_inplace(scratch_T_ptr, o);
        }
        return log_L;
    } else
        return 0;
}

template<typename dtype>
void get_log_Ref_K_i(
        np_array<dtype> log_lagrangian_mult, np_array<dtype> biased_conf_energies,
        np_array<int> count_matrices, np_array<int> state_counts,
        int n_therm_states, int n_conf_states, np_array<dtype> scratch_M, np_array<dtype> log_R_K_i
#ifdef TRAMMBAR
        ,
    double *therm_energies, int *equilibrium_therm_state_counts,
    double overcounting_factor
#endif
) {
    // TODO: define get_pointer_from_array() function for this shit
    py::buffer_info log_lagrangian_mult_buf = log_lagrangian_mult.request();
    py::buffer_info biased_conf_energies_buf = biased_conf_energies.request();
    py::buffer_info count_matrices_buf = count_matrices.request();
    py::buffer_info state_counts_buf = state_counts.request();
    py::buffer_info scratch_M_buf = scratch_M.request();
    py::buffer_info log_R_K_i_buf = log_R_K_i.request();

    dtype *log_lagrangian_mult_ptr = (dtype *) log_lagrangian_mult_buf.ptr;
    dtype *biased_conf_energies_ptr = (dtype *) biased_conf_energies_buf.ptr;
    int *count_matrices_ptr = (int *) count_matrices_buf.ptr;
    int *state_counts_ptr = (int *) state_counts_buf.ptr;
    dtype *scratch_M_ptr = (dtype *) scratch_M_buf.ptr;
    dtype *log_R_K_i_ptr = (dtype *) log_R_K_i_buf.ptr;

    int i, j, K, o;
    int Ki, Kj, KM, KMM;
    int Ci, CK, CKij, CKji, NC;
    dtype divisor, R_addon;

    for (K = 0; K < n_therm_states; ++K) {
        KM = K * n_conf_states;
        KMM = KM * n_conf_states;
        for (i = 0; i < n_conf_states; ++i) {
            Ki = KM + i;
            if (0 == state_counts_ptr[Ki]) /* applying Hao's speed-up recomendation */
            {
                log_R_K_i_ptr[Ki] = -INFINITY;
                continue;
            }
            Ci = 0;
            o = 0;
            for (j = 0; j < n_conf_states; ++j) {
                CKij = count_matrices_ptr[KMM + i * n_conf_states + j];
                CKji = count_matrices_ptr[KMM + j * n_conf_states + i];
                Ci += CKji;
                /* special case: most variables cancel out, here */
                if (i == j) {
                    scratch_M_ptr[o] = (0 == CKij) ? THERMOTOOLS_TRAM_LOG_PRIOR : log(
                            THERMOTOOLS_TRAM_PRIOR + (dtype) CKij);
                    scratch_M_ptr[o++] += biased_conf_energies_ptr[Ki];
                    continue;
                }
                CK = CKij + CKji;
                /* special case */
                if (0 == CK) continue;
                /* regular case */
                Kj = KM + j;
                divisor = _logsumexp_pair(
                        log_lagrangian_mult_ptr[Kj] - biased_conf_energies_ptr[Ki],
                        log_lagrangian_mult_ptr[Ki] - biased_conf_energies_ptr[Kj]);
                scratch_M_ptr[o++] = log((dtype) CK) + log_lagrangian_mult_ptr[Kj] - divisor;
            }
            NC = state_counts_ptr[Ki] - Ci;
            R_addon = (0 < NC) ? log((dtype) NC) + biased_conf_energies_ptr[Ki] : -INFINITY; /* IGNORE PRIOR */
            log_R_K_i_ptr[Ki] = _logsumexp_pair(_logsumexp_sort_kahan_inplace(scratch_M_ptr, o), R_addon);
        }
    }

#ifdef TRAMMBAR
    if(equilibrium_therm_state_counts && therm_energies)
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
                    log_R_K_i[Ki] = _logsumexp_pair(log_R_K_i[Ki], log(equilibrium_therm_state_counts[K]) + therm_energies[K]);
                }
            }
        }
    }
#endif
}

template<typename dtype>
void get_conf_energies(
        np_array<dtype> bias_energy_sequence, np_array<int> state_sequence, int seq_length, np_array<dtype> log_R_K_i,
        int n_therm_states, int n_conf_states, np_array<dtype> scratch_T, np_array<dtype> conf_energies) {
    py::buffer_info bias_energy_sequence_buf = bias_energy_sequence.request();
    py::buffer_info state_sequence_buf = state_sequence.request();
    py::buffer_info log_R_K_i_buf = log_R_K_i.request();
    py::buffer_info scratch_T_buf = scratch_T.request();
    py::buffer_info conf_energies_buf = conf_energies.request();

    dtype *bias_energy_sequence_ptr = (dtype *) bias_energy_sequence_buf.ptr;
    int *state_sequence_ptr = (int *) state_sequence_buf.ptr;
    dtype *log_R_K_i_ptr = (dtype *) log_R_K_i_buf.ptr;
    dtype *scratch_T_ptr = (dtype *) scratch_T_buf.ptr;
    dtype *conf_energies_ptr = (dtype *) conf_energies_buf.ptr;

    int i, K, x, o;
    double divisor;
    /* assume that conf_energies was set to INF by the caller on the first call */
    for (x = 0; x < seq_length; ++x) {
        i = state_sequence_ptr[x];
        if (i < 0) continue;
        o = 0;
        for (K = 0; K < n_therm_states; ++K) {
            if (-INFINITY == log_R_K_i_ptr[K * n_conf_states + i]) continue;
            scratch_T_ptr[o++] =
                    log_R_K_i_ptr[K * n_conf_states + i] - bias_energy_sequence_ptr[x * n_therm_states + K];
        }
        divisor = _logsumexp_sort_kahan_inplace(scratch_T_ptr, o);
        conf_energies_ptr[i] = -_logsumexp_pair(-conf_energies_ptr[i], -divisor);
    }
}

template<typename dtype>
void get_therm_energies(
        np_array<dtype> biased_conf_energies, int n_therm_states, int n_conf_states, np_array<dtype> scratch_M,
        np_array<dtype> therm_energies) {
    py::buffer_info biased_conf_energies_buf = biased_conf_energies.request();
    py::buffer_info scratch_M_buf = scratch_M.request();
    py::buffer_info therm_energies_buf = therm_energies.request();

    dtype *biased_conf_energies_ptr = (dtype *) biased_conf_energies_buf.ptr;
    dtype *scratch_M_ptr = (dtype *) scratch_M_buf.ptr;
    dtype *therm_energies_ptr = (dtype *) therm_energies_buf.ptr;

    int i, K;
    for (K = 0; K < n_therm_states; ++K) {
        for (i = 0; i < n_conf_states; ++i)
            scratch_M_ptr[i] = -biased_conf_energies_ptr[K * n_conf_states + i];
        therm_energies_ptr[K] = -_logsumexp_sort_kahan_inplace(scratch_M_ptr, n_conf_states);
    }
}

template<typename dtype>
void normalize(
        np_array<dtype> conf_energies, np_array<dtype> biased_conf_energies, np_array<dtype> therm_energies,
        int n_therm_states, int n_conf_states, np_array<dtype> scratch_M) {
    py::buffer_info conf_energies_buf = conf_energies.request();
    py::buffer_info biased_conf_energies_buf = biased_conf_energies.request();
    py::buffer_info therm_energies_buf = therm_energies.request();
    py::buffer_info scratch_M_buf = scratch_M.request();

    dtype *conf_energies_ptr = (dtype *) conf_energies_buf.ptr;
    dtype *biased_conf_energies_ptr = (dtype *) biased_conf_energies_buf.ptr;
    dtype *therm_energies_ptr = (dtype *) therm_energies_buf.ptr;
    dtype *scratch_M_ptr = (dtype *) scratch_M_buf.ptr;

    int i, KM = n_therm_states * n_conf_states;
    dtype f0;
    for (i = 0; i < n_conf_states; ++i)
        scratch_M_ptr[i] = -conf_energies_ptr[i];
    f0 = -_logsumexp_sort_kahan_inplace(scratch_M_ptr, n_conf_states);
    for (i = 0; i < n_conf_states; ++i)
        conf_energies_ptr[i] -= f0;
    for (i = 0; i < KM; ++i)
        biased_conf_energies_ptr[i] -= f0;
    for (i = 0; i < n_therm_states; ++i)
        therm_energies_ptr[i] -= f0;
}

template<typename dtype>
void estimate_transition_matrix(
        np_array<dtype> log_lagrangian_mult, np_array<dtype> conf_energies, np_array<int> count_matrix,
        int n_conf_states, np_array<dtype> scratch_M, np_array<dtype> transition_matrix) {
    py::buffer_info log_lagrangian_mult_buf = log_lagrangian_mult.request();
    py::buffer_info conf_energies_buf = conf_energies.request();
    py::buffer_info count_matrix_buf = count_matrix.request();
    py::buffer_info scratch_M_buf = scratch_M.request();
    py::buffer_info transition_matrix_buf = transition_matrix.request();

    dtype *log_lagrangian_mult_ptr = (dtype *) log_lagrangian_mult_buf.ptr;
    dtype *conf_energies_ptr = (dtype *) conf_energies_buf.ptr;
    int *count_matrix_ptr = (int *) count_matrix_buf.ptr;
    dtype *scratch_M_ptr = (dtype *) scratch_M_buf.ptr;
    dtype *transition_matrix_ptr = (dtype *) transition_matrix_buf.ptr;

    int i, j;
    int ij, ji;
    int C;
    dtype divisor, max_sum;
    dtype *sum;
    sum = scratch_M_ptr;
    for (i = 0; i < n_conf_states; ++i) {
        sum[i] = 0.0;
        for (j = 0; j < n_conf_states; ++j) {
            ij = i * n_conf_states + j;
            ji = j * n_conf_states + i;
            transition_matrix_ptr[ij] = 0.0;
            C = count_matrix_ptr[ij] + count_matrix_ptr[ji];
            /* special case: this element is zero */
            if (0 == C) continue;
            if (i == j) {
                /* special case: diagonal element */
                transition_matrix_ptr[ij] = 0.5 * C * exp(-log_lagrangian_mult_ptr[i]);
            } else {
                /* regular case */
                divisor = _logsumexp_pair(
                        log_lagrangian_mult_ptr[j] - conf_energies_ptr[i],
                        log_lagrangian_mult_ptr[i] - conf_energies_ptr[j]);
                transition_matrix_ptr[ij] = C * exp(-(conf_energies_ptr[j] + divisor));
            }
            sum[i] += transition_matrix_ptr[ij];
        }
    }
    /* normalize T matrix */ /* TODO: unify with util._renormalize_transition_matrix? */
    max_sum = 0;
    for (i = 0; i < n_conf_states; ++i) if (sum[i] > max_sum) max_sum = sum[i];
    if (max_sum == 0) max_sum = 1.0; /* completely empty T matrix -> generate Id matrix */
    for (i = 0; i < n_conf_states; ++i) {
        for (j = 0; j < n_conf_states; ++j) {
            if (i == j) {
                transition_matrix_ptr[i * n_conf_states + i] =
                        (transition_matrix_ptr[i * n_conf_states + i] + max_sum - sum[i]) / max_sum;
                if (0 == transition_matrix_ptr[i * n_conf_states + i] && 0 < count_matrix_ptr[i * n_conf_states + i])
                    fprintf(stderr, "# Warning: zero diagonal element T[%d,%d] with non-zero counts.\n", i, i);
            } else {
                transition_matrix_ptr[i * n_conf_states + j] = transition_matrix_ptr[i * n_conf_states + j] / max_sum;
            }
        }
    }

}

/* TRAM log-likelihood that comes from the terms containing discrete quantities */
template<typename dtype>
dtype discrete_log_likelihood_lower_bound(
        np_array<dtype> log_lagrangian_mult, np_array<dtype> biased_conf_energies,
        np_array<int> count_matrices, np_array<int> state_counts,
        int n_therm_states, int n_conf_states, np_array<dtype> scratch_M, np_array<dtype> scratch_MM
#ifdef TRAMMBAR
        ,
    double *therm_energies, int *equilibrium_therm_state_counts,
    double overcounting_factor
#endif
) {
    py::buffer_info log_lagrangian_mult_buf = log_lagrangian_mult.request();
    py::buffer_info biased_conf_energies_buf = biased_conf_energies.request();
    py::buffer_info count_matrices_buf = count_matrices.request();
    py::buffer_info state_counts_buf = state_counts.request();
    py::buffer_info scratch_M_buf = scratch_M.request();
    py::buffer_info scratch_MM_buf = scratch_MM.request();

    dtype *log_lagrangian_mult_ptr = (dtype *) log_lagrangian_mult_buf.ptr;
    dtype *biased_conf_energies_ptr = (dtype *) biased_conf_energies_buf.ptr;
    int *count_matrices_ptr = (int *) count_matrices_buf.ptr;
    int *state_counts_ptr = (int *) state_counts_buf.ptr;
    dtype *scratch_M_ptr = (dtype *) scratch_M_buf.ptr;
    dtype *scratch_MM_ptr = (dtype *) scratch_MM_buf.ptr;

    dtype a, b;
    int K, i, j;
    int KM, KMM, Ki;
    int CKij;
    dtype *T_ij;

    /* \sum_{i,j,k}c_{ij}^{(k)}\log p_{ij}^{(k)} */
    a = 0;
    T_ij = scratch_MM_ptr;
    for (K = 0; K < n_therm_states; ++K) {
        KM = K * n_conf_states;
        KMM = KM * n_conf_states;
        estimate_transition_matrix<dtype>(
                log_lagrangian_mult(KM), biased_conf_energies(KM), count_matrices(KMM),
                n_conf_states, scratch_M, scratch_MM);
        for (i = 0; i < n_conf_states; ++i) {
            for (j = 0; j < n_conf_states; ++j) {
                CKij = count_matrices_ptr[KMM + i * n_conf_states + j];
                if (0 == CKij) continue;
                if (i == j) {
                    a += ((dtype) CKij + THERMOTOOLS_TRAM_PRIOR) * log(T_ij[i * n_conf_states + j]);
                } else {
                    a += CKij * log(T_ij[i * n_conf_states + j]);
                }
            }
        }
    }

    /* \sum_{i,k}N_{i}^{(k)}f_{i}^{(k)} */
    b = 0;
    for (K = 0; K < n_therm_states; ++K) {
        KM = K * n_conf_states;
        for (i = 0; i < n_conf_states; ++i) {
            Ki = KM + i;
            if (state_counts_ptr[Ki] > 0)
                b += (state_counts_ptr[Ki] + THERMOTOOLS_TRAM_PRIOR) * biased_conf_energies_ptr[Ki];
        }
    }

#ifdef TRAMMBAR
    a *= overcounting_factor;
    b *= overcounting_factor;

    /* \sum_k N_{eq}^{(k)}f^{(k)}*/
    if(equilibrium_therm_state_counts && therm_energies) {
        for(K=0; K<n_therm_states; ++K) {
            if(0 < equilibrium_therm_state_counts[K])
                b += equilibrium_therm_state_counts[K] * therm_energies[K];
        }
    }
#endif

    return a + b;
}

template<typename dtype>
void get_pointwise_unbiased_free_energies(
        int k, np_array<dtype> bias_energy_sequence, np_array<dtype> therm_energies, np_array<int> state_sequence,
        int seq_length, np_array<dtype> log_R_K_i, int n_therm_states, int n_conf_states,
        np_array<dtype> scratch_T, np_array<dtype> pointwise_unbiased_free_energies) {
    py::buffer_info bias_energy_sequence_buf = bias_energy_sequence.request();
    py::buffer_info therm_energies_buf = therm_energies.request();
    py::buffer_info state_sequence_buf = state_sequence.request();
    py::buffer_info log_R_K_i_buf = log_R_K_i.request();
    py::buffer_info scratch_T_buf = scratch_T.request();
    py::buffer_info pointwise_unbiased_free_energies_buf = pointwise_unbiased_free_energies.request();

    dtype *bias_energy_sequence_ptr = (dtype *) bias_energy_sequence_buf.ptr;
    dtype *therm_energies_ptr = (dtype *) therm_energies_buf.ptr;
    int *state_sequence_ptr = (int *) state_sequence_buf.ptr;
    dtype *log_R_K_i_ptr = (dtype *) log_R_K_i_buf.ptr;
    dtype *scratch_T_ptr = (dtype *) scratch_T_buf.ptr;
    dtype *pointwise_unbiased_free_energies_ptr = (dtype *) pointwise_unbiased_free_energies_buf.ptr;

    int L, o, i, x;
    dtype log_divisor;

    for (x = 0; x < seq_length; ++x) {
        i = state_sequence_ptr[x];
        if (i < 0) {
            pointwise_unbiased_free_energies_ptr[x] = INFINITY;
            continue;
        }
        o = 0;
        for (L = 0; L < n_therm_states; ++L) {
            if (-INFINITY == log_R_K_i_ptr[L * n_conf_states + i]) continue;
            scratch_T_ptr[o++] =
                    log_R_K_i_ptr[L * n_conf_states + i] - bias_energy_sequence_ptr[x * n_therm_states + L];
        }
        log_divisor = _logsumexp_sort_kahan_inplace(scratch_T_ptr, o);
        if (k == -1)
            pointwise_unbiased_free_energies_ptr[x] = log_divisor;
        else
            pointwise_unbiased_free_energies_ptr[x] =
                    bias_energy_sequence_ptr[x * n_therm_states + k] + log_divisor - therm_energies_ptr[k];
    }
}

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
    ln_avg1 = _logsumexp_sort_kahan_inplace(scratch_ptr, L1);
    for (i = 0; i < L1; i++) {
        scratch_ptr[i] = db_JI_ptr[i] > 0 ? 0 : db_JI_ptr[i];
    }
    ln_avg2 = _logsumexp_sort_kahan_inplace(scratch_ptr, L2);
    return ln_avg2 - ln_avg1;
}