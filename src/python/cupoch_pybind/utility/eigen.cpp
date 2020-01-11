#include "cupoch_pybind/cupoch_pybind.h"
#include <thrust/host_vector.h>

namespace pybind11 {

template <typename Vector,
          typename holder_type = std::unique_ptr<Vector>,
          typename... Args>
py::class_<Vector, holder_type> bind_vector_without_repr(
        py::module &m, std::string const &name, Args &&... args) {
    // hack function to disable __repr__ for the convenient function
    // bind_vector()
    using Class_ = py::class_<Vector, holder_type>;
    Class_ cl(m, name.c_str(), std::forward<Args>(args)...);
    cl.def(py::init<>());
    cl.def("__bool__", [](const Vector &v) -> bool { return !v.empty(); },
           "Check whether the list is nonempty");
    cl.def("__len__", &Vector::size);
    return cl;
}

template <typename EigenVector>
thrust::host_vector<EigenVector> py_array_to_vectors_float(
        py::array_t<float, py::array::c_style | py::array::forcecast> array) {
    size_t eigen_vector_size = EigenVector::SizeAtCompileTime;
    if (array.ndim() != 2 || array.shape(1) != eigen_vector_size) {
        throw py::cast_error();
    }
    thrust::host_vector<EigenVector> eigen_vectors(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        // The EigenVector here must be a float-typed eigen vector, since only
        // cupoch::Vector3dVector binds to py_array_to_vectors_float.
        // Therefore, we can use the memory map directly.
        eigen_vectors[i] = Eigen::Map<EigenVector>(&array_unchecked(i, 0));
    }
    return eigen_vectors;
}

template <typename EigenVector>
thrust::host_vector<EigenVector> py_array_to_vectors_int(
        py::array_t<int, py::array::c_style | py::array::forcecast> array) {
    size_t eigen_vector_size = EigenVector::SizeAtCompileTime;
    if (array.ndim() != 2 || array.shape(1) != eigen_vector_size) {
        throw py::cast_error();
    }
    thrust::host_vector<EigenVector> eigen_vectors(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        eigen_vectors[i] = Eigen::Map<EigenVector>(&array_unchecked(i, 0));
    }
    return eigen_vectors;
}

template <typename EigenVector,
          typename EigenAllocator = Eigen::aligned_allocator<EigenVector>>
thrust::host_vector<EigenVector, EigenAllocator>
py_array_to_vectors_int_eigen_allocator(
        py::array_t<int, py::array::c_style | py::array::forcecast> array) {
    size_t eigen_vector_size = EigenVector::SizeAtCompileTime;
    if (array.ndim() != 2 || array.shape(1) != eigen_vector_size) {
        throw py::cast_error();
    }
    thrust::host_vector<EigenVector, EigenAllocator> eigen_vectors(array.shape(0));
    auto array_unchecked = array.mutable_unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        eigen_vectors[i] = Eigen::Map<EigenVector>(&array_unchecked(i, 0));
    }
    return eigen_vectors;
}

}

namespace {

template <typename Scalar,
          typename Vector = thrust::host_vector<Scalar>,
          typename holder_type = std::unique_ptr<Vector>>
py::class_<Vector, holder_type> pybind_eigen_vector_of_scalar(
        py::module &m, const std::string &bind_name) {
    auto vec = py::bind_vector<thrust::host_vector<Scalar>>(m, bind_name,
                                                    py::buffer_protocol());
    vec.def_buffer([](thrust::host_vector<Scalar> &v) -> py::buffer_info {
        return py::buffer_info(v.data(), sizeof(Scalar),
                               py::format_descriptor<Scalar>::format(), 1,
                               {v.size()}, {sizeof(Scalar)});
    });
    vec.def("__copy__",
            [](thrust::host_vector<Scalar> &v) { return thrust::host_vector<Scalar>(v); });
    vec.def("__deepcopy__", [](thrust::host_vector<Scalar> &v, py::dict &memo) {
        return thrust::host_vector<Scalar>(v);
    });
    return vec;
}

template <typename EigenVector,
          typename Vector = thrust::host_vector<EigenVector>,
          typename holder_type = std::unique_ptr<Vector>,
          typename InitFunc>
py::class_<Vector, holder_type> pybind_eigen_vector_of_vector(
        py::module &m,
        const std::string &bind_name,
        const std::string &repr_name,
        InitFunc init_func) {
    typedef typename EigenVector::Scalar Scalar;
    auto vec = py::bind_vector_without_repr<thrust::host_vector<EigenVector>>(
            m, bind_name, py::buffer_protocol());
    vec.def(py::init(init_func));
    vec.def_buffer([](thrust::host_vector<EigenVector> &v) -> py::buffer_info {
        size_t rows = EigenVector::RowsAtCompileTime;
        return py::buffer_info(v.data(), sizeof(Scalar),
                               py::format_descriptor<Scalar>::format(), 2,
                               {v.size(), rows},
                               {sizeof(EigenVector), sizeof(Scalar)});
    });
    vec.def("__repr__", [repr_name](const thrust::host_vector<EigenVector> &v) {
        return repr_name + std::string(" with ") + std::to_string(v.size()) +
               std::string(" elements.\n") +
               std::string("Use numpy.asarray() to access data.");
    });
    vec.def("__copy__", [](thrust::host_vector<EigenVector> &v) {
        return thrust::host_vector<EigenVector>(v);
    });
    vec.def("__deepcopy__", [](thrust::host_vector<EigenVector> &v, py::dict &memo) {
        return thrust::host_vector<EigenVector>(v);
    });

    // py::detail must be after custom constructor
    using Class_ = py::class_<Vector, std::unique_ptr<Vector>>;
    py::detail::vector_if_copy_constructible<Vector, Class_>(vec);
    py::detail::vector_if_equal_operator<Vector, Class_>(vec);
    py::detail::vector_modifiers<Vector, Class_>(vec);
    py::detail::vector_accessor<Vector, Class_>(vec);

    return vec;
}

template <typename EigenVector,
          typename EigenAllocator = Eigen::aligned_allocator<EigenVector>,
          typename Vector = thrust::host_vector<EigenVector, EigenAllocator>,
          typename holder_type = std::unique_ptr<Vector>,
          typename InitFunc>
py::class_<Vector, holder_type> pybind_eigen_vector_of_vector_eigen_allocator(
        py::module &m,
        const std::string &bind_name,
        const std::string &repr_name,
        InitFunc init_func) {
    typedef typename EigenVector::Scalar Scalar;
    auto vec = py::bind_vector_without_repr<
            thrust::host_vector<EigenVector, EigenAllocator>>(m, bind_name,
                                                      py::buffer_protocol());
    vec.def(py::init(init_func));
    vec.def_buffer(
            [](thrust::host_vector<EigenVector, EigenAllocator> &v) -> py::buffer_info {
                size_t rows = EigenVector::RowsAtCompileTime;
                return py::buffer_info(v.data(), sizeof(Scalar),
                                       py::format_descriptor<Scalar>::format(),
                                       2, {v.size(), rows},
                                       {sizeof(EigenVector), sizeof(Scalar)});
            });
    vec.def("__repr__",
            [repr_name](const thrust::host_vector<EigenVector, EigenAllocator> &v) {
                return repr_name + std::string(" with ") +
                       std::to_string(v.size()) + std::string(" elements.\n") +
                       std::string("Use numpy.asarray() to access data.");
            });
    vec.def("__copy__", [](thrust::host_vector<EigenVector, EigenAllocator> &v) {
        return thrust::host_vector<EigenVector, EigenAllocator>(v);
    });
    vec.def("__deepcopy__",
            [](thrust::host_vector<EigenVector, EigenAllocator> &v, py::dict &memo) {
                return thrust::host_vector<EigenVector, EigenAllocator>(v);
            });

    // py::detail must be after custom constructor
    using Class_ = py::class_<Vector, std::unique_ptr<Vector>>;
    py::detail::vector_if_copy_constructible<Vector, Class_>(vec);
    py::detail::vector_if_equal_operator<Vector, Class_>(vec);
    py::detail::vector_modifiers<Vector, Class_>(vec);
    py::detail::vector_accessor<Vector, Class_>(vec);

    return vec;
}

template <typename EigenMatrix,
          typename EigenAllocator = Eigen::aligned_allocator<EigenMatrix>,
          typename Vector = thrust::host_vector<EigenMatrix, EigenAllocator>,
          typename holder_type = std::unique_ptr<Vector>>
py::class_<Vector, holder_type> pybind_eigen_vector_of_matrix(
        py::module &m,
        const std::string &bind_name,
        const std::string &repr_name) {
    typedef typename EigenMatrix::Scalar Scalar;
    auto vec = py::bind_vector_without_repr<
            thrust::host_vector<EigenMatrix, EigenAllocator>>(m, bind_name,
                                                      py::buffer_protocol());
    vec.def_buffer(
            [](thrust::host_vector<EigenMatrix, EigenAllocator> &v) -> py::buffer_info {
                // We use this function to bind Eigen default matrix.
                // Thus they are all column major.
                size_t rows = EigenMatrix::RowsAtCompileTime;
                size_t cols = EigenMatrix::ColsAtCompileTime;
                return py::buffer_info(v.data(), sizeof(Scalar),
                                       py::format_descriptor<Scalar>::format(),
                                       3, {v.size(), rows, cols},
                                       {sizeof(EigenMatrix), sizeof(Scalar),
                                        sizeof(Scalar) * rows});
            });
    vec.def("__repr__",
            [repr_name](const thrust::host_vector<EigenMatrix, EigenAllocator> &v) {
                return repr_name + std::string(" with ") +
                       std::to_string(v.size()) + std::string(" elements.\n") +
                       std::string("Use numpy.asarray() to access data.");
            });
    vec.def("__copy__", [](thrust::host_vector<EigenMatrix, EigenAllocator> &v) {
        return thrust::host_vector<EigenMatrix, EigenAllocator>(v);
    });
    vec.def("__deepcopy__",
            [](thrust::host_vector<EigenMatrix, EigenAllocator> &v, py::dict &memo) {
                return thrust::host_vector<EigenMatrix, EigenAllocator>(v);
            });

    // py::detail must be after custom constructor
    using Class_ = py::class_<Vector, std::unique_ptr<Vector>>;
    py::detail::vector_if_copy_constructible<Vector, Class_>(vec);
    py::detail::vector_if_equal_operator<Vector, Class_>(vec);
    py::detail::vector_modifiers<Vector, Class_>(vec);
    py::detail::vector_accessor<Vector, Class_>(vec);

    return vec;
}

}  // unnamed namespace

void pybind_eigen(py::module &m) {
    py::handle static_property =
            py::handle((PyObject*)py::detail::get_internals().static_property_type);

    auto intvector = pybind_eigen_vector_of_scalar<int>(m, "IntVector");
    intvector.attr("__doc__") = static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return R"(Convert int32 numpy array of shape ``(n,)`` to Cupoch format.)";
            }),
            py::none(), py::none(), "");

    auto floatvector =
            pybind_eigen_vector_of_scalar<float>(m, "FloatVector");
    floatvector.attr("__doc__") = static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return R"(Convert float32 numpy array of shape ``(n,)`` to Cupoch format.)";
            }),
            py::none(), py::none(), "");

    auto vector3fvector = pybind_eigen_vector_of_vector<Eigen::Vector3f>(
            m, "Vector3fVector", "thrust::host_vector<Eigen::Vector3f>",
            py::py_array_to_vectors_float<Eigen::Vector3f>);
    vector3fvector.attr("__doc__") = static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return R"(Convert float32 numpy array of shape ``(n, 3)`` to Cupoch format..
Example usage
.. code-block:: python
    import cupoch
    import numpy as np
    pcd = cupoch.geometry.PointCloud()
    np_points = np.random.rand(100, 3)
    # From numpy to Cupoch
    pcd.points = cupoch.utility.Vector3fVector(np_points)
    # From Cupoch to numpy
    np_points = np.asarray(pcd.points)
)";
            }),
            py::none(), py::none(), "");

    auto vector3ivector = pybind_eigen_vector_of_vector<Eigen::Vector3i>(
            m, "Vector3iVector", "thrust::host_vector<Eigen::Vector3i>",
            py::py_array_to_vectors_int<Eigen::Vector3i>);
    vector3ivector.attr("__doc__") = static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return R"(Convert int32 numpy array of shape ``(n, 3)`` to Cupoch format..
Example usage
.. code-block:: python
    import cupoch
    import numpy as np
    # Example mesh
    # x, y coordinates:
    # [0: (-1, 2)]__________[1: (1, 2)]
    #             \        /\
    #              \  (0) /  \
    #               \    / (1)\
    #                \  /      \
    #      [2: (0, 0)]\/________\[3: (2, 0)]
    #
    # z coordinate: 0
    mesh = cupoch.geometry.TriangleMesh()
    np_vertices = np.array([[-1, 2, 0],
                            [1, 2, 0],
                            [0, 0, 0],
                            [2, 0, 0]])
    np_triangles = np.array([[0, 2, 1],
                             [1, 2, 3]]).astype(np.int32)
    mesh.vertices = cupoch.Vector3dVector(np_vertices)
    # From numpy to Cupoch
    mesh.triangles = cupoch.Vector3iVector(np_triangles)
    # From Cupoch to numpy
    np_triangles = np.asarray(mesh.triangles)
)";
            }),
            py::none(), py::none(), "");

    auto vector2ivector = pybind_eigen_vector_of_vector<Eigen::Vector2i>(
            m, "Vector2iVector", "thrust::host_vector<Eigen::Vector2i>",
            py::py_array_to_vectors_int<Eigen::Vector2i>);
    vector2ivector.attr("__doc__") = static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Convert int32 numpy array of shape ``(n, 2)`` to "
                       "Cupoch format.";
            }),
            py::none(), py::none(), "");

    auto matrix4fvector = pybind_eigen_vector_of_matrix<Eigen::Matrix4f>(
            m, "Matrix4fVector", "thrust::host_vector<Eigen::Matrix4f>");
    matrix4fvector.attr("__doc__") = static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Convert float32 numpy array of shape ``(n, 4, 4)`` to "
                       "Cupoch format.";
            }),
            py::none(), py::none(), "");

    auto vector4ivector = pybind_eigen_vector_of_vector_eigen_allocator<
            Eigen::Vector4i>(
            m, "Vector4iVector", "thrust::host_vector<Eigen::Vector4i>",
            py::py_array_to_vectors_int_eigen_allocator<Eigen::Vector4i>);
    vector4ivector.attr("__doc__") = static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Convert int numpy array of shape ``(n, 4)`` to "
                       "Cupoch format.";
            }),
            py::none(), py::none(), "");
}