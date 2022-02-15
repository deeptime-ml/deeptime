//
// Created by Maaike on 13/12/2021.
//
#pragma once

#include <deeptime/common.h>
#include "common.h"

namespace deeptime::markov::tram {

// a trajectory fragment consists of three indices. Fragment[0] is the trajectory index, fragment[1] the start index of
// the fragment within a trajectory, and fragment[2] the end index of the fragment (exclusive).
using Fragment = std::tuple<StateIndex, std::int32_t, std::int32_t>;
using Fragments = std::vector<Fragment>;

std::vector<Fragments> findTrajectoryFragmentIndices(const TTrajs &ttrajs, std::int32_t nThermStates) {

    std::vector<Fragments> fragments(nThermStates);

    for (std::size_t i = 0; i < ttrajs.size(); ++i) {
        // final index of the trajectory
        auto * begin = ttrajs[i].data();
        auto * end = begin + ttrajs[i].size();

        // first and last indices of the fragment
        auto * first = begin;

        // replica-exchange swap point
        auto * last = end;

        StateIndex thermState;

        while(first < end - 1) {
            thermState = *first;
            // look for the first occurrence of a different therm. state index.
            last = std::find_if_not(first, end, [thermState](auto x) { return x == thermState;});

            // trajectories of length one are not trajectories, they are a replica exchange swap points.
            // The swap point is the start index of the trajectory, but originates from a different therm. state.
            // e.g. [0, 0, 0, 1, 0, 0, 0] contains fragments [(0, 0, 3), (0, 3, 7)], both belonging to therm. state 0.
            if (last - first == 1) {
                thermState = *(first + 1);
                last = std::find_if_not(first + 1, end, [thermState](auto x) { return x == thermState;});
            }

            // save the indices as a trajectory fragment.
            fragments[thermState].emplace_back(i, first - begin, last - begin);

            // start next search from end of this fragment
            first = last;
        }
    }

    return fragments;
}

}
