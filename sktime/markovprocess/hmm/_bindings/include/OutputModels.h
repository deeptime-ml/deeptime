//
// Created by mho on 2/3/20.
//

#pragma once

namespace hmm {
namespace output_models {

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

}

}
}
