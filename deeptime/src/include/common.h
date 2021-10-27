#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template<typename dtype>
using np_array = py::array_t<dtype, py::array::c_style | py::array::forcecast>;
template<typename dtype>
using np_array_nfc = py::array_t<dtype, py::array::c_style>;

namespace detail {
template<typename T1, typename... T>
struct variadic_first {
    /**
     * type of the first element of a variadic type tuple
     */
    using type = typename std::decay<T1>::type;
};

}

template<typename T, typename D>
bool arraySameShape(const np_array<T>& lhs, const np_array<D>& rhs) {
    if(lhs.ndim() != rhs.ndim()) {
        return false;
    }
    for(decltype(lhs.ndim()) d = 0; d < lhs.ndim(); ++d) {
        if(lhs.shape(d) != rhs.shape(d)) return false;
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

namespace dt {
namespace constants {
template<typename dtype>
constexpr dtype pi() { return 3.141592653589793238462643383279502884e+00; }
}
}

template<std::size_t Dims>
class Index {
    static_assert(Dims > 0, "Dims has to be > 0");
public:
    /**
     * Type that holds the dimensions of the index grid
     */
    using GridDims = std::array<std::uint32_t, Dims>;
    /**
     * The value type, inherited from GridDims::value_type
     */
    using value_type = typename GridDims::value_type;

    /**
     * Constructs an empty index object of specified dimensionality. Not of much use, really.
     */
    Index() : _size(), n_elems(0) {}

    /**
     * Constructs an index object with a number of size_t arguments that must coincide with the number of dimensions,
     * specifying the grid.
     * @tparam Args the argument types, must all be size_t
     * @param args the arguments
     */
     Index(std::array<std::uint32_t, Dims> size) : _size(std::move(size)) {
        n_elems = std::accumulate(_size.begin(), _size.end(), 1u, std::multiplies<value_type>());
     }

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
    template<typename T>
    constexpr value_type operator[](T N) const {
        return _size[N];
    }

    /**
     * map Dims-dimensional index to 1D index
     * @tparam Ix the d-dimensional index template param type
     * @param ix the d-dimensional index
     * @return the 1D index
     */
    template<typename... Ix>
    constexpr value_type operator()(Ix &&... ix) const {
        static_assert(sizeof...(ix) == Dims, "wrong input dim");
        std::array<typename detail::variadic_first<Ix...>::type, Dims> indices{std::forward<Ix>(ix)...};
        // std::array<value_type, Dims> indices{std::forward<Ix>(ix)...}; // require ix to be unsigned?
        return index(indices);
    }

    /**
     * map Dims-dimensional array to 1D index
     * @param indices the Dims-dimensional index
     * @return the 1D index
     */
    template<typename Arr>
    value_type index(const Arr &indices) const {
        std::size_t result = 0;
        auto prefactor = n_elems / _size[0];
        for (std::size_t d = 0; d < Dims - 1; ++d) {
            result += prefactor * indices[d];
            prefactor /= _size[d + 1];
        }
        result += indices[Dims - 1];
        return static_cast<value_type>(result);
    }

    /**
     * Inverse mapping 1D index to Dims-dimensional tuple
     * @param idx
     * @return
     */
    GridDims inverse(std::size_t idx) const {
        GridDims result;
        auto prefactor = n_elems / _size[0];
        for(std::size_t d = 0; d < Dims-1; ++d) {
            auto x = std::floor(idx / prefactor);
            result[d] = x;
            idx -= x * prefactor;
            prefactor /= _size[d+1];
        }
        result[Dims-1] = idx;
        return result;
    }

private:
    GridDims _size;
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
