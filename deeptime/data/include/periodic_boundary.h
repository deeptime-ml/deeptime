#pragma once

#include <algorithm>
#include <array>
#include <iterator>

template<int D>
struct BoundaryConditions {
    static constexpr int DIM = D;

    template<typename It>
    static constexpr auto pbc(
        It arr, const std::array<typename std::iterator_traits<It>::value_type, DIM> &vmin,
        const std::array<typename std::iterator_traits<It>::value_type, DIM> &vmax
    ) {
        std::array<typename std::iterator_traits<It>::value_type, DIM> out;
        std::copy(arr, arr+DIM, out.begin());
        for (int d = 0; d < DIM; ++d, ++arr) {
            auto diam = vmax[d] - vmin[d];
            while (out[d] >= vmax[d]) out[d] -= diam;
            while (out[d] < vmin[d]) out[d] += diam;
        }
        return out;
    }

};
