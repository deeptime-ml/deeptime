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

    template<pybind11::ssize_t Dims>
    auto unchecked() const {
        return NpArrayMock<T>(*this);
    }

    template<pybind11::ssize_t Dims>
    auto mutable_unchecked() {
        return NpArrayMock<T>(*this);
    }

    template<typename... Ix>
    const T& operator()(Ix... ix) const {

    }

    template<typename... Ix>
    T& operator()(Ix... ix) {

    }

private:
    pybind11::array::ShapeContainer _shape;
    std::shared_ptr<T*> _data;
};

}
}
