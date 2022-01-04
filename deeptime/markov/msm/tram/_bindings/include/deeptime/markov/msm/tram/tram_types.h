//
// Created by Maaike on 14/12/2021.
//

#pragma once
#include "deeptime/common.h"

namespace deeptime::markov::tram{

using DTraj = np_array<std::int32_t>;
using DTrajs = std::vector<DTraj>;

template<typename dtype>
using BiasMatrix = np_array_nfc<dtype>;

template<typename dtype>
using BiasMatrices = std::vector<BiasMatrix<dtype>>;

using StateIndex = py::ssize_t;
}
