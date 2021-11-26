#include <catch2/catch.hpp>

#include "common.h"

TEST_CASE("Normalize yields discrete probability distributions", "[common]") {
    std::vector<float> v(1000, 0);
    std::iota(begin(v), end(v), 1);
    normalize(begin(v), end(v));
    auto sum = std::accumulate(begin(v), end(v), 0.);
    REQUIRE(sum == Approx(1.f));
}
