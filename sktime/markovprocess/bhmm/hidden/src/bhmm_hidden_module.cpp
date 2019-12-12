//
// Created by mho on 12/12/19.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace pybind11::literals;

namespace py = pybind11;

template<typename dtype>
using np_array = py::array_t<dtype, py::array::c_style>;

template<typename dtype>
py::object forward(const np_array<dtype> &arrA, const np_array<dtype> &arrpobs, const np_array<dtype> &arrpi,
                   np_array<dtype> arralpha, std::size_t T, std::size_t N) {
    auto alpha = arralpha.mutable_data();
    auto A = arrA.data();
    auto pobs = arrpobs.data();
    auto pi = arrpi.data();
    dtype sum, logprob, scaling;
    {
        py::gil_scoped_release gil;

        // first alpha and scaling factors
        scaling = 0.0;
        for (std::size_t i = 0; i < N; i++) {
            alpha[i] = pi[i] * pobs[i];
            scaling += alpha[i];
        }

        // initialize likelihood
        logprob = std::log(scaling);

        // scale first alpha
        if (scaling != 0) {
            for (std::size_t i = 0; i < N; i++) {
                alpha[i] /= scaling;
            }
        }

        // iterate trajectory
        for (std::size_t t = 0; t < T - 1; t++) {
            scaling = 0.0;
            // compute new alpha and scaling
            for (std::size_t j = 0; j < N; j++) {
                sum = 0.0;
                for (std::size_t i = 0; i < N; i++) {
                    sum += alpha[t * N + i] * A[i * N + j];
                }
                alpha[(t + 1) * N + j] = sum * pobs[(t + 1) * N + j];
                scaling += alpha[(t + 1) * N + j];
            }
            // scale this row
            if (scaling != 0) {
                for (std::size_t j = 0; j < N; j++) {
                    alpha[(t + 1) * N + j] /= scaling;
                }
            }

            // update likelihood
            logprob += std::log(scaling);
        }
    }

    return py::make_tuple(logprob, arralpha);
}

template<typename dtype>
py::object backward(const np_array<dtype> &arrA, const np_array<dtype> &arrPobs, np_array<dtype> arrBeta,
                    std::size_t T, std::size_t N) {
    auto A = arrA.data();
    auto pobs = arrPobs.data();
    auto beta = arrBeta.mutable_data();
    {
        py::gil_scoped_release gil;

        std::size_t i, j, t;
        dtype sum, scaling;

        // first beta and scaling factors
        scaling = 0.0;
        for (i = 0; i < N; i++)
        {
            beta[(T-1)*N+i] = 1.0;
            scaling += beta[(T-1)*N+i];
        }

        // scale first beta
        for (i = 0; i < N; i++)
            beta[(T-1)*N+i] /= scaling;

        // iterate trajectory
        for (t = T-1; t >= 1; t--)
        {
            scaling = 0.0;
            // compute new beta and scaling
            for (i = 0; i < N; i++)
            {
                sum = 0.0;
                for (j = 0; j < N; j++)
                {
                    sum += A[i*N+j] * pobs[t * N + j] * beta[t * N + j];
                }
                beta[(t - 1) * N + i] = sum;
                scaling += sum;
            }
            // scale this row
            if (scaling != 0)
                for (j = 0; j < N; j++)
                    beta[(t - 1) * N + j] /= scaling;
        }
    }
    return arrBeta;
}

PYBIND11_MODULE(_bhmm_hidden_bindings, m) {
    m.def("forward", &forward<float>, "A"_a, "pobs"_a, "pi"_a, "alpha"_a, "T"_a, "N"_a);
    m.def("forward", &forward<double>, "A"_a, "pobs"_a, "pi"_a, "alpha"_a, "T"_a, "N"_a);
    m.def("backward", &backward<float>, "A"_a, "pobs"_a, "beta"_a, "T"_a, "N"_a);
    m.def("backward", &backward<double>, "A"_a, "pobs"_a, "beta"_a, "T"_a, "N"_a);
}
