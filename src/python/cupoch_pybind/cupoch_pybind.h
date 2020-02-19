#pragma once

#include <pybind11/detail/internals.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>  // Include first to suppress compiler warnings
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "cupoch_pybind/device_vector_wrapper.h"

namespace py = pybind11;
using namespace py::literals;

// some helper functions
namespace pybind11 {
namespace detail {

template <typename Type, typename Alloc> struct type_caster<thrust::host_vector<Type, Alloc>>
    : list_caster<thrust::host_vector<Type, Alloc>, Type> {};

template <typename T, typename Class_>
void bind_default_constructor(Class_ &cl) {
    cl.def(py::init([]() { return new T(); }), "Default constructor");
}

template <typename T, typename Class_>
void bind_copy_functions(Class_ &cl) {
    cl.def(py::init([](const T &cp) { return new T(cp); }), "Copy constructor");
    cl.def("__copy__", [](T &v) { return T(v); });
    cl.def("__deepcopy__", [](T &v, py::dict &memo) { return T(v); });
}

inline void bind_device_vector_wrapper(py::module &m) {
    py::class_<cupoch::wrapper::device_vector_wrapper<Eigen::Vector3f>> dvv3f(m, "device_vector_vector3f");
    dvv3f.def("cpu", &cupoch::wrapper::device_vector_wrapper<Eigen::Vector3f>::cpu);
    py::class_<cupoch::wrapper::device_vector_wrapper<Eigen::Vector2f>> dvv2f(m, "device_vector_vector2f");
    dvv2f.def("cpu", &cupoch::wrapper::device_vector_wrapper<Eigen::Vector2f>::cpu);
    py::class_<cupoch::wrapper::device_vector_wrapper<Eigen::Vector3i>> dvv3i(m, "device_vector_vector3i");
    dvv3i.def("cpu", &cupoch::wrapper::device_vector_wrapper<Eigen::Vector3i>::cpu);
    py::class_<cupoch::wrapper::device_vector_wrapper<Eigen::Vector2i>> dvv2i(m, "device_vector_vector2i");
    dvv2i.def("cpu", &cupoch::wrapper::device_vector_wrapper<Eigen::Vector2i>::cpu);
    py::class_<cupoch::wrapper::device_vector_wrapper<float>> dvf(m, "device_vector_float");
    dvf.def("cpu", &cupoch::wrapper::device_vector_wrapper<float>::cpu);
    py::class_<cupoch::wrapper::device_vector_wrapper<int>> dvi(m, "device_vector_int");
    dvi.def("cpu", &cupoch::wrapper::device_vector_wrapper<int>::cpu);
    py::class_<cupoch::wrapper::device_vector_wrapper<size_t>> dvs(m, "device_vector_size_t");
    dvs.def("cpu", &cupoch::wrapper::device_vector_wrapper<size_t>::cpu);
}

}  // namespace detail
}  // namespace pybind11

PYBIND11_MAKE_OPAQUE(thrust::host_vector<int, std::allocator<int>>);
PYBIND11_MAKE_OPAQUE(thrust::host_vector<float, std::allocator<float>>);
PYBIND11_MAKE_OPAQUE(thrust::host_vector<Eigen::Vector3f, std::allocator<Eigen::Vector3f>>);