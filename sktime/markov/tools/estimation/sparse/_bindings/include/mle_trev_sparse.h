//
// Created by mho on 8/6/20.
//

#pragma once

#include "common.h"

template<typename dtype>
int mle_trev_sparse(np_array_nfc<dtype> &TArr, const np_array_nfc<dtype> &CCtArr,
                    const np_array<int> &iIndicesArr, const np_array<int> &jIndicesArr,
                    std::size_t nData, const np_array_nfc<dtype> &sumCArr, const std::size_t dim,
                    const dtype maxerr, const std::size_t maxiter,
                    np_array_nfc<dtype> &mu, dtype muEps) {
    py::gil_scoped_release gil;

    std::unique_ptr<dtype[]> sum_x{new dtype[dim]};
    std::unique_ptr<dtype[]> sum_x_new{new dtype[dim]};

    auto sumC = sumCArr.template unchecked<1>();
    auto CCt = CCtArr.template unchecked<1>();
    auto iIndices = iIndicesArr.template unchecked<1>();
    auto jIndices = jIndicesArr.template unchecked<1>();

    auto T = TArr.template mutable_unchecked<1>();

    /* ckeck sum_C */
    for (std::size_t i = 0; i < dim; ++i) {
        if (sumC(i) == 0) {
            throw std::invalid_argument("Some row and corresponding column of the count matrix C have zero counts.");
        }
    }

    {
        // initialize sum_x_new
        std::fill(sum_x_new.get(), sum_x_new.get() + dim, 0);
        dtype x_norm{0};
        for (std::size_t t = 0; t < nData; t++) {
            auto j = jIndices(t);
            auto CCt_ij = CCt(t);
            sum_x_new[j] += CCt_ij;
            x_norm += CCt_ij;
        }
        std::transform(sum_x_new.get(), sum_x_new.get() + dim, sum_x_new.get(),
                       [x_norm](auto elem) { return elem / x_norm; });
    }


    /* iterate */
    std::size_t iteration = 0;
    dtype rel_err;
    do {
        /* swap buffers */
        std::swap(sum_x, sum_x_new);

        {
            /* update sum_x */
            std::fill(sum_x_new.get(), sum_x_new.get() + dim, 0);
            for (std::size_t t = 0; t < nData; t++) {
                auto i = iIndices(t);
                auto j = jIndices(t);
                auto CCt_ij = CCt(t);
                auto value = CCt_ij / (sumC(i) / sum_x[i] + sumC(j) / sum_x[j]);
                sum_x_new[j] += value;
            }
            for (std::size_t i = 0; i < dim; ++i) {
                if (sum_x_new[i] == 0 || std::isnan(sum_x_new[i])) {
                    throw std::logic_error("The update of the stationary distribution produced zero or NaN.");
                }
            }
        }

        {
            /* normalize sum_x */
            auto xNorm = std::accumulate(sum_x_new.get(), sum_x_new.get() + dim, static_cast<dtype>(0));
            for (std::size_t i = 0; i < dim; i++) {
                sum_x_new[i] /= xNorm;
                if (sum_x_new[i] <= muEps) {
                    throw std::runtime_error("Stationary distribution contains entries smaller than "
                                             + std::to_string(muEps) + " during iteration");
                }
            }
        }


        iteration += 1;
        rel_err = util::relativeError(dim, sum_x.get(), sum_x_new.get());
    } while (rel_err > maxerr && iteration < maxiter);

    {
        // calculate X
        std::fill(sum_x.get(), sum_x.get() + dim, 0);
        for (std::size_t t = 0; t < nData; t++) {
            auto i = iIndices(t);
            auto j = jIndices(t);
            auto CCt_ij = CCt(t);
            T(t) = CCt_ij / (sumC(i) / sum_x_new[i] + sumC(j) / sum_x_new[j]);
            sum_x[i] += T(t);  // update sum with X_ij
        }
    }

    // normalize to T
    for (std::size_t t = 0; t < nData; t++) {
        auto i = iIndices(t);
        T(t) /= sum_x[i];
    }

    std::copy(sum_x_new.get(), sum_x_new.get() + dim, mu.mutable_data());

    if (iteration == maxiter) {
        return -5;
    } else {
        return 0;
    }

}

template<typename dtype>
int mle_trev_given_pi_sparse(np_array_nfc<dtype> &TunnormalizedArr, const np_array_nfc<dtype> &CCtArr,
                             const np_array<int> &iIndicesArr, const np_array<int> &jIndicesArr,
                             const std::size_t nData, const np_array_nfc<dtype> &muArr, const std::size_t len_mu,
                             const dtype maxerr, const std::size_t maxiter) {
    py::gil_scoped_release gil;

    std::unique_ptr<dtype[]> lam{new dtype[len_mu]};
    std::unique_ptr<dtype[]> lam_new{new dtype[len_mu]};

    auto CCt = CCtArr.template unchecked<1>();
    auto iIndices = iIndicesArr.template unchecked<1>();
    auto jIndices = jIndicesArr.template unchecked<1>();
    auto T = TunnormalizedArr.template mutable_unchecked<1>();
    auto mu = muArr.template unchecked<1>();

    // check mu
    for (std::size_t i = 0; i < len_mu; ++i) {
        if (mu(i) == 0) {
            throw std::invalid_argument("Some element of pi is zero.");
        }
    }

    {
        // initialize lambdas
        std::fill(lam_new.get(), lam_new.get() + len_mu, 0);

        for (std::size_t t = 0; t < nData; ++t) {
            auto i = iIndices(t);
            auto j = jIndices(t);
            if (i < j) continue;
            lam_new[i] += static_cast<dtype>(0.5) * CCt(t);
            if (i != j) {
                lam_new[j] += static_cast<dtype>(0.5) * CCt(t);
            }
        }
        for (std::size_t i = 0; i < len_mu; ++i) {
            if (lam_new[i] == 0) {
                throw std::invalid_argument("Some row and corresponding column of C have zero counts.");
            }
        }
    }

    /* iterate lambdas */
    std::size_t iteration = 0;
    dtype dsq{0};
    do {
        /* swap buffers */
        std::swap(lam, lam_new);
        std::fill(lam_new.get(), lam_new.get() + len_mu, 0);

        for (std::size_t t = 0; t < nData; t++) {
            auto i = iIndices(t);
            auto j = jIndices(t);
            if (i < j) {
                continue;
            }

            auto CCt_ij = CCt(t);
            if (CCt_ij == 0) {
                throw std::logic_error("Encountered zero in CCt. Should not happen!");
            }

            lam_new[i] += CCt_ij / ((mu(i) * lam[j]) / (mu(j) * lam[i]) + static_cast<dtype>(1));
            if (i != j) {
                lam_new[j] += CCt_ij / ((mu(j) * lam[i]) / (mu(i) * lam[j]) + static_cast<dtype>(1));
            }
        }
        for (std::size_t i = 0; i < len_mu; i++) {
            if (std::isnan(lam_new[i])) {
                throw std::runtime_error("The update of the Lagrange multipliers produced NaN.");
            }
        }

        iteration += 1;
        dsq = util::distsq(len_mu, lam.get(), lam_new.get());
    } while (dsq > maxerr * maxerr && iteration < maxiter);

    /* calculate T */
    for (std::size_t t = 0; t < nData; t++) {
        auto i = iIndices(t);
        auto j = jIndices(t);
        if (i == j) {
            T(t) = 0; // handle normalization later
        } else {
            auto CCt_ij = CCt(t);
            T(t) = CCt_ij / (lam_new[i] + lam_new[j] * mu(i) / mu(j));
        }
    }

    if (iteration == maxiter) {
        return -5;
    } else {
        return 0;
    }

}
