//
// Created by marscher on 4/3/17.
//

#ifndef PYEMMA_CLUSTERING_H
#define PYEMMA_CLUSTERING_H

#include <cstdlib>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "metric_base.h"

namespace py = pybind11;

template <typename dtype>
class ClusteringBase {

public:
    enum MetricType {
        EUCLIDEAN
    };

    using np_array = py::array_t<dtype, py::array::c_style | py::array::forcecast>;

    ClusteringBase(const std::string& metric_s) {
        if (metric_s == "euclidean") {
            typedef euclidean_metric<dtype> eucl;
            metric = std::unique_ptr<eucl>(new eucl());
            _metric_type = MetricType::EUCLIDEAN;
        } else {
            throw std::invalid_argument("metric is not of {'euclidean'}");
        }
    }

    virtual ~ClusteringBase()= default;
    ClusteringBase(const ClusteringBase&) = delete;
    ClusteringBase&operator=(const ClusteringBase&) = delete;
    ClusteringBase(ClusteringBase&&) noexcept = default;
    ClusteringBase&operator=(ClusteringBase&&) noexcept = default;

    std::unique_ptr<metric_base<dtype>> metric;

    py::array_t<int> assign_chunk_to_centers(const py::array_t<dtype, py::array::c_style>& chunk,
                                             const py::array_t<dtype, py::array::c_style>& centers,
                                             unsigned int n_threads) const {
        return metric->assign_chunk_to_centers(chunk, centers, n_threads);
    }

private:
    MetricType _metric_type;
};



#endif //PYEMMA_CLUSTERING_H
