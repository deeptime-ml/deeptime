#include <catch2/catch.hpp>
#include <pybind11/embed.h>
#include <deeptime/common.h>

TEST_CASE("Normalize yields discrete probability distributions", "[common]") {
    std::vector<float> v(1000, 0);
    std::iota(begin(v), end(v), 1);
    deeptime::normalize(begin(v), end(v));
    auto sum = std::accumulate(begin(v), end(v), 0.);
    REQUIRE(sum == Approx(1.f));
}

TEST_CASE("Index object", "[common]") {
    auto ix = deeptime::Index<5>::make_index({3, 3, 3, 3 ,3});
    REQUIRE(ix(0, 0, 0, 0, 0) == 0);
    REQUIRE(ix(0, 0, 0, 0, 1) == 1);
    REQUIRE(ix(0, 0, 0, 1, 0) == 3);
}

TEST_CASE("Swap np array", "[common]") {
    pybind11::scoped_interpreter guard{};
    auto fast_calc = py::module_::import("numpy");
    py::print("Hello, World!");
    deeptime::np_array<int> arr1 {{3, 3}};
    deeptime::np_array<int> arr2 {{3, 3}};
    arr1.mutable_at(2, 1) = 10;
    arr2.mutable_at(2, 1) = 5;
    REQUIRE(arr1.at(2, 1) == 10);
    REQUIRE(arr2.at(2, 1) == 5);

    auto ix = deeptime::Index<2>::make_index(arr1.shape(), arr1.shape() + arr1.ndim());
    auto* p1 = arr1.data();
    auto* p2 = arr2.data();
    REQUIRE(p1[ix(2, 1)] == 10);
    REQUIRE(p2[ix(2, 1)] == 5);

    std::swap(p1, p2);
    REQUIRE(p1[ix(2, 1)] == 5);
    REQUIRE(p2[ix(2, 1)] == 10);
}
