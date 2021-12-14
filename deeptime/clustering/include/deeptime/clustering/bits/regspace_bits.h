//
// Created by mho on 12/13/21.
//

#pragma once

namespace deeptime::clustering::regspace {

template<typename Metric, typename T>
inline void cluster(const np_array_nfc<T> &chunk, py::list& py_centers, T dmin, std::size_t maxClusters, int n_threads) {

    // this checks for ndim == 2
    if(chunk.ndim() != 2) {
        throw std::invalid_argument("Input chunk must be 2-dimensional but "
                                    "was " + std::to_string(chunk.ndim()) + "-dimensional.");
    }

    auto N_frames = static_cast<std::size_t>(chunk.shape(0));
    auto dim = static_cast<std::size_t>(chunk.shape(1));
    auto data = chunk.data();

    auto N_centers = py_centers.size();
    #if defined(USE_OPENMP)
    omp_set_num_threads(n_threads);
    #endif
    std::vector<np_array<T>> npCenters;
    npCenters.reserve(N_centers);
    for(std::size_t i = 0; i < N_centers; ++i) {
        npCenters.push_back(py_centers[i].cast<np_array<T>>());
    }

    // do the clustering
    for (auto i = 0U; i < N_frames; ++i) {
        auto mindist = std::numeric_limits<T>::max();
        #pragma omp parallel for reduction(min:mindist)
        for (auto j = 0U; j < N_centers; ++j) {
            auto point = npCenters.at(j).data();
            auto d = Metric::template compute(data + i*dim, point, dim);
            if (d < mindist) mindist = d;
        }
        if (mindist > dmin) {
            if (N_centers + 1 > maxClusters) {
                throw MaxCentersReachedException(
                        "Maximum number of cluster centers reached. Consider increasing max_clusters "
                        "or choose a larger minimum distance, dmin.");
            }
            // add newly found center
            std::vector<size_t> shape = {1, dim};
            np_array<T> new_center(shape, nullptr);
            std::memcpy(new_center.mutable_data(), data+i*dim, sizeof(T) * dim);

            py_centers.append(new_center);
            npCenters.push_back(new_center);
            N_centers++;
        }
    }
}

}
