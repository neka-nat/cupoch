#pragma once

#include "cupoch_pybind/cupoch_pybind.h"

namespace pybind11 {
namespace detail {

template <>
struct type_caster<Eigen::Vector3i> {
    using value_conv = make_caster<int>;

public:
    bool load(handle src, bool convert) {
        if (!isinstance<sequence>(src)) return false;
        auto l = reinterpret_borrow<sequence>(src);
        if (l.size() != 3) return false;
        size_t ctr = 0;
        for (auto it : l) {
            value_conv conv;
            if (!conv.load(it, convert)) return false;
            value[ctr++] = cast_op<int &&>(std::move(conv));
        }
        return true;
    }

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        tuple t(src.size());
        size_t index = 0;
        for (auto &&value : src) {
            auto value_ = reinterpret_steal<object>(
                    value_conv::cast(forward_like<T>(value), policy, parent));
            if (!value_) return handle();
            PyTuple_SET_ITEM(t.ptr(), (ssize_t)index++,
                             value_.release().ptr());  // steals a reference
        }
        return t.release();
    }

    PYBIND11_TYPE_CASTER(Eigen::Vector3i, _("Eigen::Vector3i"));
};

}  // namespace detail
}  // namespace pybind11