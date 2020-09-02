/**
 * Copyright (c) 2020 Neka-Nat
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
**/
#include "cupoch_pybind/dl_converter.h"

#include <Python.h>

#include "cupoch/utility/dl_converter.h"

namespace cupoch {
namespace dlpack {

template <typename T>
py::capsule ToDLpackCapsule(
        utility::device_vector<T> &src) {
    void const *managed_tensor = utility::ToDLPack(src);

    return py::capsule(managed_tensor, "dltensor", [](::PyObject *obj) {
        auto *ptr = ::PyCapsule_GetPointer(obj, "dltensor");
        if (ptr != nullptr) {
            auto *m_tsr = static_cast<::DLManagedTensor *>(ptr);
            m_tsr->deleter(m_tsr);
        } else {
            ::PyErr_Clear();
        }
    });
}

template <typename T>
void FromDLpackCapsule(
        py::capsule dlpack, utility::device_vector<T> &dst) {
    auto obj = py::cast<py::object>(dlpack);
    ::DLManagedTensor *managed_tensor =
            (::DLManagedTensor *)::PyCapsule_GetPointer(obj.ptr(), "dltensor");
    utility::FromDLPack<typename T::Scalar, T::SizeAtCompileTime>(managed_tensor, dst);
}

}
}