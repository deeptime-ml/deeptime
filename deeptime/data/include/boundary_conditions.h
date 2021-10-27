//
// Created by mho on 5/25/21.
//

#pragma once

#include <algorithm>
#include <array>
#include <iterator>
#include <tuple>

namespace deeptime::data {

template<typename State, int D, typename vmin_type, typename vmax_type, bool... periodic_dim>
struct BoundaryConditions {
    static_assert(sizeof...(periodic_dim) == D || sizeof...(periodic_dim) == 0);
    static constexpr int DIM = D;
    static constexpr std::array<bool, DIM> apply_pbc {{periodic_dim...}};

    template<std::size_t d, typename T>
    static constexpr T apply_impl_d(T x) noexcept {
        if constexpr(apply_pbc[d]) {
            auto vmax = std::tuple_element_t<d, vmax_type>::num / std::tuple_element_t<d, vmax_type>::den;
            auto vmin = std::tuple_element_t<d, vmin_type>::num / std::tuple_element_t<d, vmin_type>::den;
            auto diam = vmax - vmin;
            while (x >= vmax) x -= diam;
            while (x < vmin) x += diam;
        }
        return x;
    }

    template<std::size_t... I>
    static constexpr State apply_impl(State in, std::index_sequence<I...>) noexcept {
        ((in[I] = apply_impl_d<I>(in[I])), ...);
        return in;
    }

    template<typename Indices = std::make_index_sequence<DIM>>
    static constexpr auto apply(State in) noexcept {
        if constexpr(sizeof...(periodic_dim) == 0) {
            return in;
        } else {
            return apply_impl(in, Indices{});
        }
    }
};

template<typename State>
using NoPeriodicBoundaryConditions = BoundaryConditions<State, 0, void, void>;

}
