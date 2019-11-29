#pragma once

#include <pybind11/detail/internals.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>  // Include first to suppress compiler warnings
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <thrust/host_vector.h>
#include "cupoch/utility/eigen.h"

namespace py = pybind11;
using namespace py::literals;

// some helper functions
namespace pybind11 {
namespace detail {

template <typename Type, typename Alloc> struct type_caster<thrust::host_vector<Type, Alloc> >
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

}  // namespace detail
}  // namespace pybind11

// PYBIND11_MAKE_OPAQUE(thrust::host_vector<int>);
// PYBIND11_MAKE_OPAQUE(thrust::host_vector<float>);
// PYBIND11_MAKE_OPAQUE(thrust::host_vector<Eigen::Vector3f_u>);