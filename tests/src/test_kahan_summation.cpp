#include <catch2/catch.hpp>
#include <pybind11/embed.h>
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


TEST_CASE("kdot") {
    py::scoped_interpreter guard;
    SECTION("shape mismatch") {
        auto a = np_array_nfc<double>({3, 2});
        auto b = np_array_nfc<double>({2, 3});
        try {
            deeptime::numeric::kahan::kdot(a, b);
        }
        catch (std::invalid_argument const &err) {
            REQUIRE(err.what() == std::string("Shape mismatch, A.shape[1] must match B.shape[0]."));
        }
        catch (...) {
            FAIL("Unexpected exception was thrown.");
        }
    }
    SECTION("check correct output shape") {
       auto a = np_array_nfc<double>({1, 2});
       auto b = np_array_nfc<double>({2, 3});

       auto res = deeptime::numeric::kahan::kdot(a, b);
       REQUIRE(res.shape(0) == a.shape(0));
       REQUIRE(res.shape(1) == b.shape(1));
   }

    SECTION("check correct output vectors") {
        auto a = np_array_nfc<double>({1,2});
        auto b = np_array_nfc<double>({2,1});

        a.mutable_at(0, 0) = std::sqrt(2.)/2;
        a.mutable_at(0, 1) = -std::sqrt(2.)/2;
        
        // some values on the unit circle with their respective cosine.
        auto x = GENERATE(0, 1, -std::sqrt(2.)/2, std::sqrt(2.)/2, -1);
        auto y = GENERATE(0, 1, -std::sqrt(2.)/2, std::sqrt(2.)/2, -1);
	
        b.mutable_at(0,0) = x;
        b.mutable_at(1,0) = y;
	auto correct = a.at(0,0) * b.at(0,0) + a.at(0, 1) * b.at(1,0);
        auto res = deeptime::numeric::kahan::kdot(a, b);
        REQUIRE(Approx(res.at(0,0)) == correct);
    }

    SECTION("check correct output matrix") {
        auto a = np_array_nfc<double>(std::vector<int>{2, 4});
        auto b = np_array_nfc<double>(std::vector<int>{4, 1});

        // fill with some random numbers (use number generator??)
        auto aBuf = a.mutable_unchecked();
        aBuf(0, 0) = -9364837;
        aBuf(0, 1) = 6354931;
        aBuf(0, 2) = 2933099;
        aBuf(0, 3) = 2962495;
        aBuf(1, 0) = 1917769;
        aBuf(1, 1) = -1596682;
        aBuf(1, 2) = 3273189;
        aBuf(1, 3) = -1714555;
        b.mutable_at(0, 0) = 0.;
        b.mutable_at(1, 0) = 1;
        b.mutable_at(2, 0) = 100;
        b.mutable_at(3, 0) = 1000000.;

        auto res = deeptime::numeric::kahan::kdot(a, b);
        REQUIRE(res.at(0, 0) == 1000000 * aBuf(0, 3) + 100 * aBuf(0, 2) + aBuf(0, 1));
        REQUIRE(res.at(1, 0) == aBuf(1, 1) + 100 * aBuf(1, 2) + 1000000 * aBuf(1, 3));
    }
}
