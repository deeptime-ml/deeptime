#include <pybind11/embed.h>
#include <catch2/catch.hpp>

#include "deeptime/markov/tram/tram.h"

TEST_CASE("TRAM Swap", "[tram]") {
    using namespace deeptime::markov::tram;

    pybind11::scoped_interpreter interpreterGuard;

    ExchangeableArray<float, 3> ex {std::vector<int>{3, 5, 7}, 33.};
    REQUIRE(ex.first()->at(1, 1, 1) == 33);
    ex.firstBuf()(1, 1, 1) = 66;
    REQUIRE(ex.first()->at(1, 1, 1) == 66);
    ex.exchange();
    REQUIRE(ex.first()->at(1, 1, 1) == 33);
    ex.exchange();
    REQUIRE(ex.first()->at(1, 1, 1) == 66);
}
