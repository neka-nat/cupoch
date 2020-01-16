#include <Python.h>
#include "cupoch_pybind/dl_converter.h"
#include "cupoch/utility/dl_converter.h"

using namespace cupoch;
using namespace cupoch::dlpack;

py::capsule cupoch::dlpack::ToDLpackCapsule(thrust::device_vector<Eigen::Vector3f>& src) {
    void const *managed_tensor = utility::ToDLPack(src);

    return py::capsule(managed_tensor, "dltensor", [](::PyObject *obj) {
        auto *ptr = ::PyCapsule_GetPointer(obj, "dltensor");
        if (ptr != nullptr) {
            auto *m_tsr = static_cast<::DLManagedTensor*>(ptr);
            m_tsr->deleter(m_tsr);
        }
        else {
            ::PyErr_Clear();
        }
    });
}

void cupoch::dlpack::FromDLpackCapsule(py::capsule dlpack, thrust::device_vector<Eigen::Vector3f>& dst) {
    auto obj = py::cast<py::object>(dlpack);
    ::DLManagedTensor* managed_tensor = (::DLManagedTensor*)::PyCapsule_GetPointer(obj.ptr(), "dltensor");
    utility::FromDLPack<float, 3>(managed_tensor, dst);
}