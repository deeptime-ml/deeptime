//
// Created by mho on 7/18/19.
//

#pragma once

#include "common.h"

py::array_t<std::int32_t> countStates(const py::list& dtrajs) {
    std::vector<std::int32_t> counts;
    for(auto dtraj : dtrajs) {
        auto npDtraj = py::cast<py::array_t<std::int32_t>>(dtraj);
        auto r = npDtraj.unchecked<1>();

        for(auto i = 0U; i < npDtraj.size(); ++i) {
            auto state = r(i);
            if(state >= 0) {
                if (static_cast<std::int64_t>(counts.size()) <= state) {
                    counts.resize(state + 1);
                }
                counts.at(state) += 1;
            }
        }
    }

    py::array_t<std::int32_t> result {std::vector<std::size_t>{counts.size()}};
    {
        auto r = result.mutable_data(0);
        for (auto i = 0U; i < counts.size(); ++i) {
            *(r+i) = counts.at(i);
        }
    }
    return result;
}

auto indexStates(const py::list& dtrajs, const py::object &pySubset) {
    auto hist = countStates(dtrajs);
    auto nStates = hist.size();

    auto isRequested = pySubset.is_none() ? std::vector<bool>(nStates, true) : std::vector<bool>(nStates, false);

    py::array_t<std::int32_t> subset;

    std::size_t subsetSize = nStates;
    if(!pySubset.is_none()) {
        subset = py::cast<py::array_t<std::int32_t>>(pySubset);

        auto max = std::max_element(subset.data(0), subset.data(0) + subset.size());
        if (*max >= nStates) {
            isRequested.resize(*max + 1);
            nStates = *max + 1;
        }

        subsetSize = subset.size();
        for(auto i = 0U; i < subsetSize; ++i) {
            isRequested.at(subset.at(i)) = true;
        }
    } else {
        subset = py::array_t<std::int32_t>(static_cast<std::size_t>(nStates));
        auto ptr = subset.mutable_data(0);
        std::iota(ptr, ptr + nStates, 0);
    }


    std::vector<std::int32_t> counts (subsetSize, 0);
    // stateToContiguousIndex[state] maps to the index of the state to a contiguous subset index
    std::vector<std::int32_t> stateToContiguousIndex (nStates, 0);

    py::list result;
    for(auto i = 0U; i < subsetSize; ++i) {
        auto state = subset.at(i);
        auto nAppend = state < hist.size() ? hist.at(state) : 0;
        py::array_t<std::int32_t> zeros {std::vector<std::size_t>{static_cast<std::size_t>(nAppend), 2}};
        result.append(std::move(zeros));
        stateToContiguousIndex[subset.at(i)] = i;
    }

    // avoid casting within the main loop
    std::vector<py::array_t<std::int32_t>> dtrajsvec;
    for(auto dtraj : dtrajs) {
        dtrajsvec.push_back(dtraj.cast<py::array_t<std::int32_t>>());
    }

    {
        auto itDtrajs = dtrajsvec.begin();
        std::size_t currentDtrajIndex = 0;
        for (; itDtrajs != dtrajsvec.end(); ++itDtrajs, ++currentDtrajIndex) {
            const auto& arr = *itDtrajs;
            auto data = arr.data(0);
            for(auto t = 0U; t < arr.size(); ++t) {
                auto state = *(data + t);
                if(state >= 0 && isRequested[state]) {
                    auto k = stateToContiguousIndex[state];
                    auto resultArray = result[k].cast<py::array_t<std::int32_t>>();
                    resultArray.mutable_at(counts[k], 0) = currentDtrajIndex;
                    resultArray.mutable_at(counts[k], 1) = t;
                    ++counts.at(k);
                }
            }
        }
    }
    return result;
}
