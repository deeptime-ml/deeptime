#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace deeptime {

template<typename dtype>
using np_array = py::array_t<dtype, py::array::c_style | py::array::forcecast>;
template<typename dtype>
using np_array_nfc = py::array_t<dtype, py::array::c_style>;

template<typename T, typename D>
bool arraySameShape(const np_array<T> &lhs, const np_array<D> &rhs) {
    if (lhs.ndim() != rhs.ndim()) {
        return false;
    }
    for (decltype(lhs.ndim()) d = 0; d < lhs.ndim(); ++d) {
        if (lhs.shape(d) != rhs.shape(d)) return false;
    }
    return true;
}

template<typename Iter1, typename Iter2>
void normalize(Iter1 begin, Iter2 end) {
    auto sum = std::accumulate(begin, end, typename std::iterator_traits<Iter1>::value_type());
    for (auto it = begin; it != end; ++it) {
        *it /= sum;
    }
}

namespace detail {
template<typename... Ix>
struct ComputeIndex {
    template<typename Strides, typename Indices = std::make_index_sequence<sizeof...(Ix)>>
    static constexpr auto compute(const Strides &strides, Ix &&... ix) {
        std::tuple<Ix...> tup(std::forward<Ix>(ix)...);
        return compute(strides, tup, Indices{});
    }

    template<typename Arr, std::size_t... I>
    static constexpr auto compute(const Arr &strides, const std::tuple<Ix...> &tup, std::index_sequence<I...>) {
        return (0 + ... + (strides[I] * std::get<I>(tup)));;
    }
};

}

template<std::size_t Dims, typename T = std::array<std::uint32_t, Dims>>
class Index {
    static_assert(Dims > 0, "Dims has to be > 0");
public:
    using GridDims = T;

    template<typename It>
    static auto make_index(It shapeBegin, It shapeEnd) {
        GridDims dims;
        std::copy(shapeBegin, shapeEnd, begin(dims));
        auto n_elems = std::accumulate(begin(dims), end(dims), 1u, std::multiplies<value_type>());

        GridDims strides;
        strides[0] = n_elems / dims[0];
        for (std::size_t d = 0; d < Dims - 1; ++d) {
            strides[d + 1] = strides[d] / dims[d + 1];
        }

        return Index<Dims, GridDims>{dims, strides, n_elems};
    }

    template<typename Container = std::initializer_list<typename GridDims::value_type>>
    static auto make_index(const Container &container) {
        return make_index(begin(container), end(container));
    }

    /**
     * The value type, inherited from GridDims::value_type
     */
    using value_type = typename GridDims::value_type;

    /**
     * Constructs an empty index object of specified dimensionality. Not of much use, really.
     */
    Index() : _size(), _cum_size(), n_elems(0) {}

    template<typename Shape>
    Index(const Shape &size)
            : _size(), n_elems(std::accumulate(begin(size), end(size), 1u, std::multiplies<value_type>())) {
        std::copy(begin(size), end(size), begin(_size));

        GridDims strides;
        strides[0] = n_elems / size[0];
        for (std::size_t d = 0; d < Dims - 1; ++d) {
            strides[d + 1] = strides[d] / size[d + 1];
        }
        _cum_size = std::move(strides);
    }

    /**
     * Constructs an index object with a number of size_t arguments that must coincide with the number of dimensions,
     * specifying the grid.
     * @tparam Args the argument types, must all be size_t
     * @param args the arguments
     */
    Index(GridDims size, GridDims strides, value_type nElems) : _size(std::move(size)), _cum_size(std::move(strides)),
                                                                n_elems(nElems) {}

    /**
     * the number of elements in this index, exactly the product of the grid dimensions
     * @return the number of elements
     */
    value_type size() const {
        return n_elems;
    }

    /**
     * the number of elements in this index
     * @return the number of elements
     */
    value_type nElements() const {
        return n_elems;
    }

    /**
     * Retrieve size of N-th axis
     * @tparam N the axis
     * @return size of N-th axis
     */
    template<int N>
    constexpr value_type get() const {
        return _size[N];
    }

    /**
     * retrieve size of N-th axis
     * @param N N
     * @return size of N-th axis
     */
    template<typename D>
    constexpr value_type operator[](D N) const {
        return _size[N];
    }

    /**
     * map Dims-dimensional index to 1D index
     * @tparam Ix the d-dimensional index template param type
     * @param ix the d-dimensional index
     * @return the 1D index
     */
    template<typename... Ix, typename Indices = std::make_index_sequence<Dims>>
    constexpr value_type operator()(Ix &&... ix) const {
        static_assert(std::size_t(sizeof...(ix)) == Dims, "wrong input dim");
        return detail::ComputeIndex<Ix...>::compute(_cum_size, std::forward<Ix>(ix)...);
    }

    /**
     * map Dims-dimensional array to 1D index
     * @param indices the Dims-dimensional index
     * @return the 1D index
     */
    template<typename Arr>
    value_type index(const Arr &indices) const {
        std::size_t result{0};
        for (std::size_t i = 0; i < Dims; ++i) {
            result += _cum_size[i] * indices[i];
        }
        return result;
    }

    /**
     * Inverse mapping 1D index to Dims-dimensional tuple
     * @param idx
     * @return
     */
    GridDims inverse(std::size_t idx) const {
        GridDims result;
        auto prefactor = n_elems / _size[0];
        for (std::size_t d = 0; d < Dims - 1; ++d) {
            auto x = std::floor(idx / prefactor);
            result[d] = x;
            idx -= x * prefactor;
            prefactor /= _size[d + 1];
        }
        result[Dims - 1] = idx;
        return result;
    }

private:
    GridDims _size;
    GridDims _cum_size;
    value_type n_elems;
};

namespace util {
template<typename dtype>
auto relativeError(const std::size_t n, const dtype *const a, const dtype *const b) -> dtype {
    auto max = static_cast<dtype>(0);
    for (auto i = 0U; i < n; i++) {
        auto sum = static_cast<dtype>(.5) * (a[i] + b[i]);
        if (sum > 0) {
            auto d = std::abs((a[i] - b[i]) / sum);
            if (d > max) {
                max = d;
            }
        }
    }
    return max;
}

template<typename dtype>
static dtype distsq(const std::size_t n, const dtype *const a, const dtype *const b) {
    dtype d = 0.0;
    #pragma omp parallel for reduction(+:d) default(none) firstprivate(n, a, b)
    for (std::size_t i = 0; i < n; i++) {
        d += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return d;
}
}
}
