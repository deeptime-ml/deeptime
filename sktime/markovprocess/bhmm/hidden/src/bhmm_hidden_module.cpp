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
                   np_array<dtype> arralpha, std::size_t T) {
    auto alpha = arralpha.mutable_data();
    auto A = arrA.data();
    auto pobs = arrpobs.data();
    auto pi = arrpi.data();

    auto N = static_cast<std::size_t>(arrA.shape(0));

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
                    std::size_t T) {
    auto A = arrA.data();
    auto pobs = arrPobs.data();
    auto beta = arrBeta.mutable_data();

    auto N = static_cast<std::size_t>(arrA.shape(0));
    {
        py::gil_scoped_release gil;

        std::size_t i, j, t;
        dtype sum, scaling;

        // first beta and scaling factors
        scaling = 0.0;
        for (i = 0; i < N; i++) {
            beta[(T - 1) * N + i] = 1.0;
            scaling += beta[(T - 1) * N + i];
        }

        // scale first beta
        for (i = 0; i < N; i++)
            beta[(T - 1) * N + i] /= scaling;

        // iterate trajectory
        for (t = T - 1; t >= 1; t--) {
            scaling = 0.0;
            // compute new beta and scaling
            for (i = 0; i < N; i++) {
                sum = 0.0;
                for (j = 0; j < N; j++) {
                    sum += A[i * N + j] * pobs[t * N + j] * beta[t * N + j];
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

template<typename dtype>
void transitionCounts(const np_array<dtype> &alpha, const np_array<dtype> &beta, const np_array<dtype> &A,
                      const np_array<dtype> &pobs, std::size_t T, np_array<dtype> &counts) {
    auto countsBuf = counts.mutable_data();
    auto alphaBuf = alpha.data();
    auto ABuf = A.data();
    auto betaBuf = beta.data();
    auto pobsBuf = pobs.data();

    auto N = static_cast<std::size_t>(A.shape(0));

    {
        py::gil_scoped_release gil;

        std::fill(countsBuf, countsBuf + counts.size(), 0.0);

        dtype sum;

        std::unique_ptr<dtype[]> tmp = std::unique_ptr<dtype[]>(new dtype[N * N]);

        for (std::size_t t = 0; t < T - 1; t++) {
            sum = 0.0;
            for (std::size_t i = 0; i < N; i++) {
                for (std::size_t j = 0; j < N; j++) {
                    tmp[i * N + j] =
                            alphaBuf[t * N + i] * ABuf[i * N + j] * pobsBuf[(t + 1) * N + j] * betaBuf[(t + 1) * N + j];
                    sum += tmp[i * N + j];
                }
            }
            for (std::size_t i = 0; i < N; i++) {
                for (std::size_t j = 0; j < N; j++) {
                    countsBuf[i * N + j] += tmp[i * N + j] / sum;
                }
            }
        }
    }
}

template<typename dtype>
np_array<int> viterbi(const np_array<dtype> &A, const np_array<dtype> &pobs, const np_array<dtype> &pi, std::size_t T) {
    auto N = static_cast<std::size_t>(A.shape(0));

    np_array<std::int32_t> path(std::vector<std::size_t>{T});
    auto pathBuf = path.mutable_data();
    auto ABuf = A.data();
    auto pobsBuf = pobs.data();
    auto piBuf = pi.data();
    {
        py::gil_scoped_release gil;

        std::fill(pathBuf, pathBuf + path.size(), 0);

        std::size_t i, j, t, maxi;
        dtype sum;
        auto vData = std::unique_ptr<dtype[]>(new dtype[N]);
        auto v = vData.get();
        auto vnextData = std::unique_ptr<dtype[]>(new dtype[N]);
        auto vnext = vnextData.get();
        auto hData = std::unique_ptr<dtype[]>(new dtype[N]);
        auto h = hData.get();
        auto ptr = std::unique_ptr<std::int32_t[]>(new std::int32_t[T * N]);

        // initialization of v
        sum = 0.0;
        for (i = 0; i < N; i++) {
            v[i] = pobsBuf[i] * piBuf[i];
            sum += v[i];
        }
        // normalize
        for (i = 0; i < N; i++) {
            v[i] /= sum;
        }

        // iteration of v
        for (t = 1; t < T; t++) {
            sum = 0.0;
            for (j = 0; j < N; j++) {
                for (i = 0; i < N; i++) {
                    h[i] = v[i] * ABuf[i * N + j];
                }
                maxi = std::distance(h, std::max_element(h, h + N));
                ptr[t * N + j] = maxi;
                vnext[j] = pobsBuf[t * N + j] * v[maxi] * ABuf[maxi * N + j];
                sum += vnext[j];
            }
            // normalize
            for (i = 0; i < N; i++) {
                vnext[i] /= sum;
            }
            // update v
            std::swap(v, vnext);
        }

        // path reconstruction
        pathBuf[T - 1] = std::distance(v, std::max_element(v, v + N));
        for (t = T - 1; t >= 1; t--) {
            pathBuf[t - 1] = ptr[t * N + pathBuf[t]];
        }


    }
    return path;
}

template<typename Iter1, typename Iter2>
void normalize(Iter1 begin, Iter2 end) {
    auto sum = std::accumulate(begin, end, typename std::iterator_traits<Iter1>::value_type());
    for(auto it = begin; it != end; ++it) {
        *it /= sum;
    }
}

template<typename T>
int _random_choice(const T* const p, const int N)
{
    double dR = (double)rand();
    double dM = (double)RAND_MAX;
    double r = dR / (dM + 1.0);
    double s = 0.0;
    int i;
    for (i = 0; i < N; i++)
    {
        s += p[i];
        if (s >= r)
        {
            return i;
        }
    }

    return -1;
}

template<typename dtype>
np_array<std::int32_t> samplePath(const np_array<dtype> &alpha, const np_array<dtype> &A, const np_array<dtype> &pobs, std::size_t T,
                int seed = -1) {
    auto N = static_cast<std::size_t>(A.shape(0));

    np_array<std::int32_t> pathArray(std::vector<std::size_t>{T});
    auto path = pathArray.mutable_data();

    auto pselPtr = std::unique_ptr<dtype[]>(new dtype[N]);
    auto psel = pselPtr.get();

    auto alphaBuf = alpha.data();
    auto ABuf = A.data();
    {
        py::gil_scoped_release gil;
        // initialize random number generator
        if (seed == -1) {
            srand(time(NULL));
        } else {
            srand(seed);
        }

        // Sample final state.
        for (std::size_t i = 0; i < N; i++) {
            psel[i] = alphaBuf[(T - 1) * N + i];
        }

        normalize(psel, psel + N);
        // Draw from this distribution.
        path[T - 1] = _random_choice(psel, N);

        // Work backwards from T-2 to 0.
        for (std::size_t t = T - 1; t >= 1; t--) {
            // Compute P(s_t = i | s_{t+1}..s_T).
            for (std::size_t i = 0; i < N; i++) {
                psel[i] = alphaBuf[(t - 1) * N + i] * ABuf[i * N + path[t]];
            }
            normalize(psel, psel + N);
            // Draw from this distribution.
            path[t - 1] = _random_choice(psel, N);
        }
    }

    return pathArray;
}

PYBIND11_MODULE(_bhmm_hidden_bindings, m) {
    m.def("forward", &forward<float>, "A"_a, "pobs"_a, "pi"_a, "alpha"_a, "T"_a);
    m.def("forward", &forward<double>, "A"_a, "pobs"_a, "pi"_a, "alpha"_a, "T"_a);
    m.def("backward", &backward<float>, "A"_a, "pobs"_a, "beta"_a, "T"_a);
    m.def("backward", &backward<double>, "A"_a, "pobs"_a, "beta"_a, "T"_a);
    m.def("transition_counts", &transitionCounts<float>, "alpha"_a, "beta"_a, "A"_a, "pobs"_a, "T"_a, "C"_a);
    m.def("transition_counts", &transitionCounts<double>, "alpha"_a, "beta"_a, "A"_a, "pobs"_a, "T"_a, "C"_a);
    m.def("viterbi", &viterbi<float>, "A"_a, "pobs"_a, "pi"_a, "T"_a);
    m.def("viterbi", &viterbi<double>, "A"_a, "pobs"_a, "pi"_a, "T"_a);
    m.def("sample_path", &samplePath<float>, "alpha"_a, "A"_a, "pobs"_a, "T"_a, "seed"_a = -1);
    m.def("sample_path", &samplePath<double>, "alpha"_a, "A"_a, "pobs"_a, "T"_a, "seed"_a = -1);
}
