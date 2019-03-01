//
// Created by marscher on 7/17/17.
//

#ifndef PYEMMA_REGSPACE_H
#define PYEMMA_REGSPACE_H

#if defined(USE_OPENMP)
#include <omp.h>
#endif

#include <Clustering.h>


class MaxCentersReachedException : public std::exception {
public:
    explicit MaxCentersReachedException(const char * m) : message{m} {}
    virtual const char * what() const noexcept override {return message.c_str();}
private:
    std::string message = "";
};

template<typename dtype>
class RegularSpaceClustering : public ClusteringBase<dtype> {
    using parent_t = ClusteringBase<dtype>;
public:
    /**
     *
     * @param dmin
     * @param max_clusters
     * @param metric
     * @param input_dimension
     */
    RegularSpaceClustering(dtype dmin, std::size_t max_clusters,
                           const std::string &metric,
                           size_t input_dimension) :
            ClusteringBase<dtype>(metric, input_dimension),
            dmin(dmin),
            max_clusters(max_clusters) {}

    /**
     * loops over all points in chunk and checks for each center if the distance is smaller than dmin,
     * if so, the point is appended to py_centers. This is done until max_centers is reached or all points have been
     * added to the list.
     * @param chunk array shape(n, d)
     * @param py_centers python list containing found centers.
     */
    void cluster(const py::array_t <dtype, py::array::c_style> &chunk, py::list& py_centers, unsigned int n_threads) {
        // this checks for ndim == 2
        const auto& data = chunk.template unchecked< 2 >();

        std::size_t N_frames = chunk.shape(0);
        std::size_t dim = chunk.shape(1);
        std::size_t N_centers = py_centers.size();
        #if defined(USE_OPENMP)
        omp_set_num_threads(n_threads);
        #endif
        // do the clustering
        for (std::size_t i = 0; i < N_frames; ++i) {
            auto mindist = std::numeric_limits<dtype>::max();
            #pragma omp parallel for reduction(min:mindist)
            for (std::size_t j = 0; j < N_centers; ++j) {
                // TODO avoid the cast in inner loop?
                auto point = py_centers[j].cast < py::array_t < dtype >> ();
                auto d = parent_t::metric.get()->compute(&data(i, 0), point.data());
                if (d < mindist) mindist = d;
            }
            if (mindist > dmin) {
                if (N_centers + 1 > max_clusters) {
                    throw MaxCentersReachedException(
                            "Maximum number of cluster centers reached. Consider increasing max_clusters "
                            "or choose a larger minimum distance, dmin.");
                }
                // add newly found center
                std::vector<size_t> shape = {1, dim};
                py::array_t<dtype> new_center(shape, nullptr);
                std::memcpy(new_center.mutable_data(), &data(i, 0), sizeof(dtype)*dim);

                py_centers.append(new_center);
                N_centers++;
            }
        }
    }

protected:
    dtype dmin;
    std::size_t max_clusters;

};

#endif //PYEMMA_REGSPACE_H
