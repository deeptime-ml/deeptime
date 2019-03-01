//
// Created by marscher on 4/3/17.
//

#ifndef PYEMMA_KMEANS_H
#define PYEMMA_KMEANS_H

#include <utility>

#include "Clustering.h"

namespace py = pybind11;


template<typename dtype>
class KMeans : public ClusteringBase<dtype> {
public:
    using parent_t = ClusteringBase<dtype>;
    using np_array = py::array_t<dtype, py::array::c_style | py::array::forcecast>;
    /**
      * array with new cluster centers, return code (0 == converged), number of iterations taken.
      */
    using cluster_res = std::tuple<np_array, int, int>;

    KMeans(unsigned int k,
           const std::string &metric,
           size_t input_dimension) : ClusteringBase<dtype>(metric, input_dimension), k(k) {}

    /**
     * performs kmeans clustering on the given data chunk, provided a list of centers.
     * @param np_chunk
     * @param np_centers
     * @param n_threads
     * @return updated centers.
     */
    np_array cluster(const np_array & /*np_chunk*/, const np_array & /*np_centers*/, int /*n_threads*/) const;

    /**
      *
      */
    cluster_res cluster_loop(const np_array & /*np_chunk*/, np_array & /*np_centers*/,
                             int /*n_threads*/, int /*max_iter*/, float /*tolerance*/,
                             py::object& /*callback*/) const;

    /**
     * evaluate the quality of the centers
     *
     * @return
     */
    dtype costFunction(const np_array & /*np_data*/, const np_array & /*np_centers*/, int /*n_threads*/) const;

    /**
     * kmeans++ initialisation
     * @param np_data
     * @param random_seed
     * @param n_threads
     * @return init centers.
     */
    np_array initCentersKMpp(const np_array& /*np_data*/, unsigned int /*random_seed*/, int /*n_threads*/,
                             py::object& /*callback*/) const;

protected:
    unsigned int k;
};

#include "bits/kmeans_bits.h"

#endif //PYEMMA_KMEANS_H
