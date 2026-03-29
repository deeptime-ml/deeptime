#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "deeptime/numeric/kahan_summation.h"

using namespace deeptime;

TEST_CASE("Kahan pairwise logsumexp", "[kahan_summation]") {

    auto a = -std::numeric_limits<double>::infinity();
    auto b = -std::numeric_limits<double>::infinity();

    SECTION("arguments both infinity") {
        auto res = deeptime::numeric::kahan::logsumexp_pair(a, b);
        REQUIRE(res == -std::numeric_limits<double>::infinity());
    }SECTION("b=-infinity yields a") {
        b = 10;
        auto res = deeptime::numeric::kahan::logsumexp_pair(a, b);
        REQUIRE(res == b);
    }SECTION("a=-infinity yields b") {
        a = 10;
        b = -std::numeric_limits<double>::infinity();
        auto res = deeptime::numeric::kahan::logsumexp_pair(a, b);
        REQUIRE(res == a);
    }SECTION("No infinities yields correct output") {
        a = 3.;
        b = 5.;
        auto res = deeptime::numeric::kahan::logsumexp_pair(a, b);
        REQUIRE(res == std::log(std::exp(a) + std::exp(b)));

        res = deeptime::numeric::kahan::logsumexp_pair(b, a);
        REQUIRE(res == std::log(std::exp(a) + std::exp(b)));
    }
}

TEST_CASE("logsumexp_kahan_inplace", "[kahan_Summation]") {
    const int len = 5;
    double neg_inf = -std::numeric_limits<double>::infinity();
    double arr[len] = {neg_inf, neg_inf, neg_inf, neg_inf, neg_inf};

    SECTION("zero length array returns -inf") {
        arr[0] = {1.};
        double res = deeptime::numeric::kahan::logsumexp_kahan_inplace(arr, arr, 100.);
        REQUIRE(res == -std::numeric_limits<double>::infinity());
    }SECTION("arraymax=-inf returns -inf") {
        double arr[1] = {1.};
        double res = deeptime::numeric::kahan::logsumexp_kahan_inplace(arr, arr + len,
                                                                       -std::numeric_limits<double>::infinity());
        REQUIRE(res == -std::numeric_limits<double>::infinity());
    }SECTION("sum is correct") {
        arr[2] = 3.;
        arr[4] = 5.;
        double res = deeptime::numeric::kahan::logsumexp_kahan_inplace(arr, arr + len, 5.);
        REQUIRE(res == std::log(std::exp(3.) + std::exp(5.)));
    }
}

TEST_CASE("logsumexp_sort_kahan_inplace", "[kahan_summation]") {
    const int len = 5;
    double neg_inf = -std::numeric_limits<double>::infinity();
    double arr[len] = {neg_inf, neg_inf, neg_inf, neg_inf, neg_inf};

    SECTION("-inf array returns -inf - overload (begin, end)") {
        double res = deeptime::numeric::kahan::logsumexp_sort_kahan_inplace(arr, arr + len);
        REQUIRE(res == -std::numeric_limits<double>::infinity());
        res = deeptime::numeric::kahan::logsumexp_sort_kahan_inplace(arr, len);
        REQUIRE(res == -std::numeric_limits<double>::infinity());
    }SECTION("zero length array yields -inf") {
        arr[0] = 3.;
        arr[1] = 5.;
        double res = deeptime::numeric::kahan::logsumexp_sort_kahan_inplace(arr, arr);
        REQUIRE(res == -std::numeric_limits<double>::infinity());

        res = deeptime::numeric::kahan::logsumexp_sort_kahan_inplace(arr, 0);
        REQUIRE(res == -std::numeric_limits<double>::infinity());
    }SECTION("Kahan logsumexp sort inplace yields correct value - low to high") {
        arr[2] = 3.;
        arr[4] = 100.;

        SECTION("overload (begin, end)") {
            double res = deeptime::numeric::kahan::logsumexp_sort_kahan_inplace(arr, arr + 5);
            REQUIRE(res == std::log(std::exp(3.) + std::exp(100.)));
        }SECTION("overload (begin, size)") {
            double res = deeptime::numeric::kahan::logsumexp_sort_kahan_inplace(arr, 5);
            REQUIRE(res == std::log(std::exp(3.) + std::exp(100.)));
        }
    }SECTION("Kahan logsumexp sort inplace yields correct value - high to low") {
        arr[4] = 3.;
        arr[2] = 100.;
        SECTION("overload (begin, end)") {
            double res = deeptime::numeric::kahan::logsumexp_sort_kahan_inplace(arr, arr + 5);
            REQUIRE(res == std::log(std::exp(3.) + std::exp(100.)));
        }SECTION("overload (begin, size)") {
            double res = deeptime::numeric::kahan::logsumexp_sort_kahan_inplace(arr, 5);
            REQUIRE(res == std::log(std::exp(3.) + std::exp(100.)));
        }
    }
    SECTION("Kahan logsumexp sort inplace does not overflow") {
        arr[4] = 3.;
        arr[2] = 10.e8;
        SECTION("overload (begin, end)") {
            REQUIRE(std::log(std::exp(arr[4]) + std::exp(arr[2])) == std::numeric_limits<double>::infinity());
            double res = deeptime::numeric::kahan::logsumexp_sort_kahan_inplace(arr, arr + 5);
            REQUIRE(res < std::numeric_limits<double>::infinity());
        }SECTION("overload (begin, size)") {
            REQUIRE(std::log(std::exp(arr[4]) + std::exp(arr[2])) == std::numeric_limits<double>::infinity());
            double res = deeptime::numeric::kahan::logsumexp_sort_kahan_inplace(arr, 5);
            REQUIRE(res < std::numeric_limits<double>::infinity());
        }
    }
}


TEST_CASE("kdot", "[kahan_summation]") {
    SECTION("check correct output vectors") {
        // a(1x2) @ b(2x1) -> c(1x1)
        double a[2] = {std::sqrt(2.) / 2, -std::sqrt(2.) / 2};
        double b[2];
        double c[1];

        auto x = GENERATE(0., 1., -std::sqrt(2.) / 2, std::sqrt(2.) / 2, -1.);
        auto y = GENERATE(0., 1., -std::sqrt(2.) / 2, std::sqrt(2.) / 2, -1.);

        b[0] = x;
        b[1] = y;
        auto correct = a[0] * b[0] + a[1] * b[1];
        deeptime::numeric::kahan::kdot_raw(a, b, c, 1, 2, 1);
        REQUIRE_THAT(c[0], Catch::Matchers::WithinRel(correct, 0.001));
    }

    SECTION("check correct output matrix") {
        // a(2x4) @ b(4x1) -> c(2x1)
        double a[8] = {-9364837, 6354931, 2933099, 2962495,
                        1917769, -1596682, 3273189, -1714555};
        double b[4] = {0., 1., 100., 1000000.};
        double c[2] = {0, 0};

        deeptime::numeric::kahan::kdot_raw(a, b, c, 2, 4, 1);
        REQUIRE(c[0] == 1000000 * 2962495. + 100 * 2933099. + 6354931.);
        REQUIRE(c[1] == -1596682. + 100 * 3273189. + 1000000 * (-1714555.));
    }
}
