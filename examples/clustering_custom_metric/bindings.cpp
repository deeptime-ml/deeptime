#include "register_clustering.h"

struct MaximumMetric {

    template<typename dtype>
    static dtype compute(const dtype* xs, const dtype* ys, std::size_t dim) {
        dtype result = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            auto d = std::abs(xs[i] - ys[i]);
            if (d > result) {
                result = d;
            }
        }
        return result;
    }

    template<typename dtype>
    static dtype compute_squared(const dtype* xs, const dtype* ys, std::size_t dim) {
        auto d = compute(xs, ys, dim);
        return d*d;
    }
};

PYBIND11_MODULE(bindings, m) {
    auto maxnormModule = m.def_submodule("maxnorm");
    deeptime::clustering::registerClusteringImplementation<MaximumMetric>(maxnormModule);
}
