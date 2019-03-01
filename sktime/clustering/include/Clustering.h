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
        EUCLIDEAN, MINRMSD
    };

    using np_array = py::array_t<dtype, py::array::c_style | py::array::forcecast>;

    ClusteringBase(const std::string& metric_s, std::size_t input_dimension) : input_dimension(input_dimension) {
        if (metric_s == "euclidean") {
            typedef euclidean_metric<dtype> eucl;
            metric = std::unique_ptr<eucl>(new eucl(input_dimension));
            _metric_type = MetricType::EUCLIDEAN;
        } else if(metric_s == "minRMSD") {
            typedef min_rmsd_metric<float> min_rmsd_t;
            metric = std::unique_ptr<min_rmsd_t>(new min_rmsd_t(input_dimension));
            _metric_type = MetricType::MINRMSD;
        } else {
            throw std::invalid_argument("metric is not of {'euclidean', 'minRMSD'}");
        }
    }

    virtual ~ClusteringBase()= default;
    ClusteringBase(const ClusteringBase&) = delete;
    ClusteringBase&operator=(const ClusteringBase&) = delete;
    ClusteringBase(ClusteringBase&&) noexcept = default;
    ClusteringBase&operator=(ClusteringBase&&) noexcept = default;

    std::unique_ptr<metric_base<dtype>> metric;
    std::size_t input_dimension;

    py::array_t<int> assign_chunk_to_centers(const py::array_t<dtype, py::array::c_style>& chunk,
                                             const py::array_t<dtype, py::array::c_style>& centers,
                                             unsigned int n_threads) const {
        return metric->assign_chunk_to_centers(chunk, centers, n_threads);
    }

    /**
     * pre-center given centers in place
     * @param centers
     */
    void precenter_centers(np_array& centers) const {
        switch (_metric_type) {
            case MetricType::MINRMSD: {
                auto ptr = dynamic_cast<min_rmsd_metric<dtype>*>(metric.get());
                ptr->precenter_centers(centers.mutable_data(0), centers.shape(0));
                break;
            }
            default: {
                throw std::runtime_error("precentering is only available for minRMSD metric.");
            };
        }
    }

private:
    MetricType _metric_type;
};



#endif //PYEMMA_CLUSTERING_H
