#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <deeptime/common.h>

TEST_CASE("Normalize yields discrete probability distributions", "[common]") {
    std::vector<float> v(1000, 0);
    std::iota(begin(v), end(v), 1);
    deeptime::normalize(begin(v), end(v));
    auto sum = std::accumulate(begin(v), end(v), 0.);
    REQUIRE_THAT( sum, Catch::Matchers::WithinRel(1.f, 0.001f));
}

TEST_CASE("Index object", "[common]") {
    auto ix = deeptime::Index<5>::make_index({3, 3, 3, 3 ,3});
    REQUIRE(ix(0, 0, 0, 0, 0) == 0);
    REQUIRE(ix(0, 0, 0, 0, 1) == 1);
    REQUIRE(ix(0, 0, 0, 1, 0) == 3);
}
