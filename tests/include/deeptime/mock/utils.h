//
// Created by mho on 11/29/21.
//

#pragma once

#include <pybind11/numpy.h>

namespace deeptime {
namespace mock {

template<typename T>
struct NpArrayMock {
    NpArrayMock(pybind11::array::ShapeContainer &&shape) : _shape(std::move(shape)) {
        _data = std::shared_ptr<T>(new T[size()]);
    }

    auto size() const {
        return std::accumulate(begin(_shape), end(_shape), 1, std::multiplies<>());
    }

private:
    pybind11::array::ShapeContainer _shape;
    std::shared_ptr<T*> _data;
};

}
}
