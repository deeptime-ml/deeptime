//
// Created by mho on 5/25/21.
//

#pragma once

#include <algorithm>
#include <array>
#include <iterator>
#include <tuple>

namespace deeptime::data {

template<typename State>
struct NoPeriodicBoundaryConditions {
    static constexpr auto apply(State in) {
        return in;
    }
};


template<typename State, int D, typename VMIN, typename VMAX, bool... periodic_dim>
struct BoundaryConditions {
    static_assert(sizeof...(periodic_dim) == D || sizeof...(periodic_dim) == 0);
    static constexpr int DIM = D;
    static constexpr std::array<bool, DIM> apply_pbc = sizeof...(periodic_dim) == 0 ? std::array<bool, DIM>{} : std::array<bool, DIM>{{periodic_dim...}};

    template<std::size_t... I>
    static constexpr State vmin_impl(std::index_sequence<I...>) {
        return {{std::tuple_element<I, VMIN>::type::num / std::tuple_element<I, VMIN>::type::den ... }};
    }

    template<typename Indices = std::make_index_sequence<DIM>>
    static constexpr State vmin() {
        return vmin_impl(Indices{});
    }

    template<std::size_t... I>
    static constexpr State vmax_impl(std::index_sequence<I...>) {
        return {{std::tuple_element<I, VMAX>::type::num / std::tuple_element<I, VMAX>::type::den ... }};
    }

    template<typename Indices = std::make_index_sequence<DIM>>
    static constexpr State vmax() {
        return vmax_impl(Indices{});
    }

    static constexpr auto apply(State in) {
        if constexpr(sizeof...(periodic_dim) == 0) {
            return in;
        } else {
            for (int d = 0; d < DIM; ++d) {
                if (apply_pbc[d]) {
                    auto diam = vmax()[d] - vmin()[d];
                    while (in[d] >= vmax()[d]) in[d] -= diam;
                    while (in[d] < vmin()[d]) in[d] += diam;
                }
            }
            return in;
        }
    }
};
}
