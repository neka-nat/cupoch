#include "cupoch/visualization/visualizer/view_parameters.h"

#include <json/json.h>
#include <Eigen/Dense>

#include "cupoch/utility/console.h"

namespace cupoch {
namespace visualization {

ViewParameters::Vector17f ViewParameters::ConvertToVector17f() {
    ViewParameters::Vector17f v;
    v(0) = field_of_view_;
    v(1) = zoom_;
    v.block<3, 1>(2, 0) = lookat_;
    v.block<3, 1>(5, 0) = up_;
    v.block<3, 1>(8, 0) = front_;
    v.block<3, 1>(11, 0) = boundingbox_min_;
    v.block<3, 1>(14, 0) = boundingbox_max_;
    return v;
}

void ViewParameters::ConvertFromVector17f(const ViewParameters::Vector17f &v) {
    field_of_view_ = v(0);
    zoom_ = v(1);
    lookat_ = v.block<3, 1>(2, 0);
    up_ = v.block<3, 1>(5, 0);
    front_ = v.block<3, 1>(8, 0);
    boundingbox_min_ = v.block<3, 1>(11, 0);
    boundingbox_max_ = v.block<3, 1>(14, 0);
}

bool ViewParameters::ConvertToJsonValue(Json::Value &value) const {
    value["field_of_view"] = field_of_view_;
    value["zoom"] = zoom_;
    if (EigenVector3fToJsonArray(lookat_, value["lookat"]) == false) {
        return false;
    }
    if (EigenVector3fToJsonArray(up_, value["up"]) == false) {
        return false;
    }
    if (EigenVector3fToJsonArray(front_, value["front"]) == false) {
        return false;
    }
    if (EigenVector3fToJsonArray(boundingbox_min_, value["boundingbox_min"]) ==
        false) {
        return false;
    }
    if (EigenVector3fToJsonArray(boundingbox_max_, value["boundingbox_max"]) ==
        false) {
        return false;
    }
    return true;
}

bool ViewParameters::ConvertFromJsonValue(const Json::Value &value) {
    if (value.isObject() == false) {
        utility::LogWarning(
                "ViewParameters read JSON failed: unsupported json format.");
        return false;
    }
    field_of_view_ = value.get("field_of_view", 60.0).asFloat();
    zoom_ = value.get("zoom", 0.7).asFloat();
    if (EigenVector3fFromJsonArray(lookat_, value["lookat"]) == false) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.");
        return false;
    }
    if (EigenVector3fFromJsonArray(up_, value["up"]) == false) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.");
        return false;
    }
    if (EigenVector3fFromJsonArray(front_, value["front"]) == false) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.");
        return false;
    }
    if (EigenVector3fFromJsonArray(boundingbox_min_,
                                   value["boundingbox_min"]) == false) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.");
        return false;
    }
    if (EigenVector3fFromJsonArray(boundingbox_max_,
                                   value["boundingbox_max"]) == false) {
        utility::LogWarning("ViewParameters read JSON failed: wrong format.");
        return false;
    }
    return true;
}

}  // namespace visualization
}  // namespace cupoch