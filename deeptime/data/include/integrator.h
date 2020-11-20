//
// Created by mho on 11/20/20.
//

#pragma once


//------------------------------------------------------------------------------
// Runge-Kutta integrator for ordinary differential equations
//------------------------------------------------------------------------------
template<typename Rhs, typename dtype, std::size_t DIM>
class RungeKutta {
public:
    RungeKutta(dtype h, size_t nSteps) : h(h), nSteps(nSteps) {}

    void eval(dtype* x, dtype *y) {
        std::copy(x, x + DIM, y); // copy initial value to y

        for (std::size_t i = 0; i < nSteps; ++i) {

            Rhs(y, k1); // compute k1 = f(y)

            for (std::size_t j = 0; j < DIM; ++j) {
                yt[j] = y[j] + h / 2 * k1[j];
            }
            Rhs(yt, k2); // compute k2 = f(y+h/2*k1)

            for(std::size_t j = 0; j < DIM; ++j) {
                yt[j] = y[j] + h / 2 * k2[j];
            }
            Rhs(yt, k3); // compute k3 = f(y+h/2*k2)

            for (std::size_t j = 0; j < DIM; ++j) {
                yt[j] = y[j] + h * k3[j];
            }
            Rhs(yt, k4); // compute k4 = f(y+h*k3)

            for (size_t j = 0; j < DIM; ++j) {
                // compute x_{k+1} = x_k + h/6*(k1 + 2*k2 + 2*k3 + k4)
                y[j] = y[j] + h / 6.0 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
            }
        }
    }

private:
    dtype h;                        ///< step size
    std::size_t nSteps;              ///< number of integration steps
    std::array<dtype, DIM> k1, k2, k3, k4, yt; ///< temporary variables
};

//------------------------------------------------------------------------------
// Euler-Maruyama integrator for stochastic differential equations
//------------------------------------------------------------------------------
template<typename T>
class EulerMaruyama {
public:
    EulerMaruyama(size_t d, T h, size_t nSteps);

    void eval(Vector<T> &x, Vector<T> &y);

    void updateNoise(Vector<T> &w);

private:
    std::default_random_engine generator_;       ///< random number generator
    std::normal_distribution<T> distribution_;   ///< normal distribution
    const T h_;                                  ///< step size of the integrator
    const T sqrt_h_;                             ///< precomputed square root of h for efficiency
    const size_t nSteps_;                        ///< number of integration steps
    Vector<T> mu_;                               ///< temporary vector
};


//------------------------------------------------------------------------------
// Runge-Kutta integrator for ordinary differential equations (impl.)
//------------------------------------------------------------------------------
template<typename T>
RungeKutta<T>::RungeKutta(ODE<T> *ode, size_t d, T h, size_t nSteps) : ode_(ode), h_(h), nSteps_(nSteps) {

}

template<typename T>
void RungeKutta<T>::eval(Vector<T> &x, Vector<T> &y) {

}

//------------------------------------------------------------------------------
// Euler-Maruyama integrator for stochastic differential equations (impl.)
//------------------------------------------------------------------------------
template<typename T>
EulerMaruyama<T>::EulerMaruyama(SDE<T> *sde, size_t d, T h, size_t nSteps)
        : sde_(sde),
          h_(h),
          nSteps_(nSteps),
          generator_(std::chrono::system_clock::now().time_since_epoch().count()),
          distribution_(0.0, 1.0),
          sqrt_h_(std::sqrt(h_)) {
    mu_.resize(d);
}

template<typename T>
void EulerMaruyama<T>::eval(Vector<T> &x, Vector<T> &y) {
    const size_t d = sde_->getDimension();

    Matrix<T> sigma(d); // sigma is required to be constant here
    for (size_t i = 0; i < d; ++i)
        sigma[i].resize(d);
    sde_->getSigma(sigma);

    Vector<T> w(d); // for the Wiener processes

    y = x; // copy initial value to y

    for (size_t i = 0; i < nSteps_; ++i) {
        sde_->f(y, mu_); // compute drift term mu
        updateNoise(w);  // evaluate Wiener processes

        for (size_t j = 0; j < d; ++j) {
            y[j] = y[j] + h_ * mu_[j];

            for (size_t k = 0; k < d; ++k)
                y[j] += sigma[j][k] * sqrt_h_ * w[k];
        }
    }
}

template<typename T>
void EulerMaruyama<T>::updateNoise(Vector<T> &w) {
    const size_t d = sde_->getDimension();
    for (size_t i = 0; i < d; ++i)
        w[i] = distribution_(generator_);
}

