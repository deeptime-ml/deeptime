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

    template<std::size_t d, typename T>
    static constexpr T apply_impl_d(T x) {
        if constexpr(apply_pbc[d]) {
            auto vmax = std::tuple_element<d, VMAX>::type::num / std::tuple_element<d, VMAX>::type::den;
            auto vmin = std::tuple_element<d, VMIN>::type::num / std::tuple_element<d, VMIN>::type::den;
            auto diam = vmax - vmin;
            while (x >= vmax) x -= diam;
            while (x < vmin) x += diam;
        }
        return x;
    }

    template<std::size_t... I>
    static constexpr State apply_impl(State in, std::index_sequence<I...>) {
        using expander = int[];
        (void) expander{0, ((void) (in[I] = apply_impl_d<I>(in[I])), 0)...};
        return in;
    }

    template<typename Indices = std::make_index_sequence<DIM>>
    static constexpr auto apply(State in) {
        if constexpr(sizeof...(periodic_dim) == 0) {
            return in;
        } else {
            return apply_impl(in, Indices{});
        }
    }
};
}
