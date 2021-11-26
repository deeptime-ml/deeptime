//
// Created by mho on 11/25/21.
//
// todo bench PBF
#include <cstddef>
#include <pybind11/embed.h>
#include "common.h"

#include <benchmark/benchmark.h>

template<typename... Ix>
struct ComputeIndex {
    template<typename Strides, typename Indices = std::make_index_sequence<sizeof...(Ix)>>
    static constexpr auto compute(const Strides &strides, Ix ... ix) {
        std::tuple<Ix...> tup(ix...);
        return compute(strides, tup, Indices{});
    }

    template<typename Arr, std::size_t... I>
    static constexpr auto compute(const Arr &strides, const std::tuple<Ix...> &tup, std::index_sequence<I...>) {
        return (0 + ... + (strides[I] * std::get<I>(tup)));;
    }
};

template <ssize_t Dim = 0, typename Strides> ssize_t ComputeIndex2(const Strides &) { return 0; }
template <ssize_t Dim = 0, typename Strides, typename... Ix>
ssize_t ComputeIndex2(const Strides &strides, ssize_t i, Ix... index) {
    return i * strides[Dim] + ComputeIndex2<Dim + 1>(strides, index...);
}


template<typename T, std::size_t Dim>
class PtrProxy {
    std::array<std::size_t, Dim> shape, strides;
    const std::byte* _data;
    // const T* _data;
public:
    PtrProxy() = default;
    template<typename P>
    PtrProxy(const P* shape, const T* data) : _data(reinterpret_cast<const std::byte*>(data)) {
        auto n_elems = std::accumulate(shape, shape+Dim, 1u, std::multiplies<>());
        strides[0] = n_elems * itemsize() / shape[0];
        for(std::size_t d = 0; d < Dim; ++d) {
            this->shape[d] = shape[d];
            if (d < Dim - 1) {
                strides[d + 1] = strides[d] / shape[d + 1];
            }
        }
    }
    constexpr static auto itemsize() { return sizeof(T); }

    template <typename... Ix>
    const T& operator()(Ix... index) const {
        static_assert(sizeof...(Ix) == Dim, "Wrong dimension");
        //return *reinterpret_cast<const T *>(_data + ComputeIndex<Ix...>::compute(strides, index...));
        return *reinterpret_cast<const T *>(_data + ComputeIndex2(strides, index...));
    }
};


template <typename T, ssize_t Dims>
class unchecked_reference {
protected:
    static constexpr bool Dynamic = Dims < 0;
    const unsigned char *data_;
    // Storing the shape & strides in local variables (i.e. these arrays) allows the compiler to
    // make large performance gains on big, nested loops, but requires compile-time dimensions
    std::conditional_t<Dynamic, const ssize_t *, std::array<ssize_t, (size_t) Dims>>
            shape_, strides_;
    const ssize_t dims_;

    friend class pybind11::array;
public:
    // Constructor for compile-time dimensions:
    template <bool Dyn = Dynamic>
    unchecked_reference(const void *data, const ssize_t *shape, const ssize_t *strides, std::enable_if_t<!Dyn, ssize_t>)
            : data_{reinterpret_cast<const unsigned char *>(data)}, dims_{Dims} {
        for (size_t i = 0; i < (size_t) dims_; i++) {
            shape_[i] = shape[i];
            strides_[i] = strides[i];
        }
    }
    // Constructor for runtime dimensions:
    template <bool Dyn = Dynamic>
    unchecked_reference(const void *data, const ssize_t *shape, const ssize_t *strides, std::enable_if_t<Dyn, ssize_t> dims)
            : data_{reinterpret_cast<const unsigned char *>(data)}, shape_{shape}, strides_{strides}, dims_{dims} {}

    /**
     * Unchecked const reference access to data at the given indices.  For a compile-time known
     * number of dimensions, this requires the correct number of arguments; for run-time
     * dimensionality, this is not checked (and so is up to the caller to use safely).
     */
    template <typename... Ix> const T &operator()(Ix... index) const {
        static_assert(ssize_t{sizeof...(Ix)} == Dims || Dynamic,
                      "Invalid number of indices for unchecked array reference");
        return *reinterpret_cast<const T *>(data_ + pybind11::detail::byte_offset_unsafe(strides_, ssize_t(index)...));
    }
    /**
     * Unchecked const reference access to data; this operator only participates if the reference
     * is to a 1-dimensional array.  When present, this is exactly equivalent to `obj(index)`.
     */
    template <ssize_t D = Dims, typename = std::enable_if_t<D == 1 || Dynamic>>
    const T &operator[](ssize_t index) const { return operator()(index); }

    /// Pointer access to the data at the given indices.
    template <typename... Ix> const T *data(Ix... ix) const { return &operator()(ssize_t(ix)...); }

    /// Returns the item size, i.e. sizeof(T)
    constexpr static ssize_t itemsize() { return sizeof(T); }

    /// Returns the shape (i.e. size) of dimension `dim`
    ssize_t shape(ssize_t dim) const { return shape_[(size_t) dim]; }

    /// Returns the number of dimensions of the array
    ssize_t ndim() const { return dims_; }

    /// Returns the total number of elements in the referenced array, i.e. the product of the shapes
    template <bool Dyn = Dynamic>
    std::enable_if_t<!Dyn, ssize_t> size() const {
        return std::accumulate(shape_.begin(), shape_.end(), (ssize_t) 1, std::multiplies<ssize_t>());
    }
    template <bool Dyn = Dynamic>
    std::enable_if_t<Dyn, ssize_t> size() const {
        return std::accumulate(shape_, shape_ + ndim(), (ssize_t) 1, std::multiplies<ssize_t>());
    }

    /// Returns the total number of bytes used by the referenced data.  Note that the actual span in
    /// memory may be larger if the referenced array has non-contiguous strides (e.g. for a slice).
    ssize_t nbytes() const {
        return size() * itemsize();
    }
};

template<typename Arr, typename Prox>
auto sumItUp(const Arr& arr, const Prox &prox) {
    double sum = 0.;
    for (py::ssize_t i = 0; i < arr.shape(0); i++) {
        for (py::ssize_t j = 0; j < arr.shape(1); j++) {
            for (py::ssize_t k = 0; k < arr.shape(2); k++) {
                sum += prox(i, j, k);
            }
        }
    }
    return sum;
}

static void BM_Direct(benchmark::State& state) {
    // Perform setup here
    py::scoped_interpreter guard {};
    std::vector<py::ssize_t> shape {300, 200, 200};
    np_array<double> array {shape};
    std::iota(array.mutable_data(), array.mutable_data() + array.size(), 0.);
    // auto ix = Index<3>::make_index(array.shape(), array.shape() + array.ndim());
    // std::vector<py::ssize_t> shape{array.shape(), array.shape() + array.ndim()};
    auto* begin = array.data();
    auto* end = array.data() + array.size();
    double sum;
    for (auto _ : state) {
        benchmark::DoNotOptimize(sum = std::accumulate(begin, end, 0.));
    }
}
BENCHMARK(BM_Direct);

static void BM_Proxy(benchmark::State& state) {
    // Perform setup here
    py::scoped_interpreter guard {};
    std::vector<py::ssize_t> shape {300, 200, 200};
    np_array<double> array {shape};
    std::iota(array.mutable_data(), array.mutable_data() + array.size(), 0.);
    // auto ix = Index<3>::make_index(array.shape(), array.shape() + array.ndim());
    // std::vector<py::ssize_t> shape{array.shape(), array.shape() + array.ndim()};
    PtrProxy<double, 3> proxy {array.shape(), array.data()};
    double sum;
    for (auto _ : state) {
        benchmark::DoNotOptimize(sum = sumItUp(array, proxy));
    }
}
BENCHMARK(BM_Proxy);


static void BM_URCopy(benchmark::State& state) {
    // Perform setup here
    py::scoped_interpreter guard {};
    std::vector<py::ssize_t> shape {300, 200, 200};
    np_array<double> array {shape};
    std::iota(array.mutable_data(), array.mutable_data() + array.size(), 0.);
    unchecked_reference<double, 3> ref {array.data(), array.shape(), array.strides(), array.ndim()};
    double sum;
    for (auto _ : state) {
        benchmark::DoNotOptimize(sum = sumItUp(array, ref));
    }
}
BENCHMARK(BM_URCopy);


static void BM_Index(benchmark::State& state) {
    // Perform setup here
    py::scoped_interpreter guard {};
    std::vector<py::ssize_t> shape {300, 200, 200};
    np_array<double> array {shape};
    std::iota(array.mutable_data(), array.mutable_data() + 300 * 200 * 200, 0.);
    // auto ix = Index<3>::make_index(array.shape(), array.shape() + array.ndim());
    // std::vector<py::ssize_t> shape{array.shape(), array.shape() + array.ndim()};
    auto ptr = array.data();
    Index<3> ix {shape};
    auto fun = [&] () {
        double sum = 0;
        for (py::ssize_t i = 0; i < array.shape(0); i++) {
            for (py::ssize_t j = 0; j < array.shape(1); j++) {
                for (py::ssize_t k = 0; k < array.shape(2); k++) {
                    sum += ptr[ix(i, j, k)];
                }
            }
        }
        return sum;
    };
    double sum ;
    for (auto _ : state) {
        benchmark::DoNotOptimize(sum = fun());
    }
}
BENCHMARK(BM_Index);

static void BM_Pybind11Unchecked(benchmark::State& state) {
    // Perform setup here
    py::scoped_interpreter guard {};
    std::vector<py::ssize_t> shape {300, 200, 200};
    np_array<double> arr {std::move(shape)};
    std::iota(arr.mutable_data(), arr.mutable_data() + arr.size(), 0.);

    np_array_nfc<double> array {arr};
    double sum;
    auto buf = array.unchecked<3>();
    for (auto _ : state) {
        benchmark::DoNotOptimize(sum = sumItUp(array, buf));
    }
}
BENCHMARK(BM_Pybind11Unchecked);

// Run the benchmark
BENCHMARK_MAIN();


/*int main() {
    py::scoped_interpreter guard {};
    py::module np = py::module::import("numpy");
    np_array<int> arr1 {{3, 3}};
    np_array<int> arr2 {{3, 3}};
    py::print("Hello, World!");
    return 0;
}
*/