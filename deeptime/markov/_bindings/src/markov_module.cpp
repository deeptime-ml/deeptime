/********************************************************************************
 * This file is part of scikit-time.                                            *
 *                                                                              *
 * Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)         *
 *                                                                              *
 * scikit-time is free software: you can redistribute it and/or modify          *
 * it under the terms of the GNU Lesser General Public License as published by  *
 * the Free Software Foundation, either version 3 of the License, or            *
 * (at your option) any later version.                                          *
 *                                                                              *
 * This program is distributed in the hope that it will be useful,              *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
 * GNU General Public License for more details.                                 *
 *                                                                              *
 * You should have received a copy of the GNU Lesser General Public License     *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.        *
 ********************************************************************************/

// author: clonker

#include "discrete_trajectories.h"
#include "simulation.h"

using namespace pybind11::literals;

PYBIND11_MODULE(_markov_bindings, m) {
    {
        auto sampleMod = m.def_submodule("sample");
        sampleMod.def("index_states", &indexStates, py::arg("dtrajs"), py::arg("subset") = py::none());
        sampleMod.def("count_states", &countStates, py::arg("dtrajs"));
    }
    {
        auto simMod = m.def_submodule("simulation");
        simMod.def("trajectory", &trajectory<float>, "N"_a, "start"_a, "P"_a, "stop"_a = py::none(), "seed"_a = -1);
        simMod.def("trajectory", &trajectory<double>, "N"_a, "start"_a, "P"_a, "stop"_a = py::none(), "seed"_a = -1);
    }
}
