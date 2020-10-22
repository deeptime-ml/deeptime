#################################################################################
# This file is part of scikit-time                                              #
#                                                                               #
# Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)          #
#                                                                               #
# scikit-time is free software: you can redistribute it and/or modify           #
# it under the terms of the GNU Lesser General Public License as published by   #
# the Free Software Foundation, either version 3 of the License, or             #
# (at your option) any later version.                                           #
#                                                                               #
# This program is distributed in the hope that it will be useful,               #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                 #
# GNU General Public License for more details.                                  #
#                                                                               #
# You should have received a copy of the GNU Lesser General Public License      #
# along with this program.  If not, see <http://www.gnu.org/licenses/>.         #
#################################################################################

project(_markov_bindings CXX)

set(SRC src/markov_module.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include ${common_includes})
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
