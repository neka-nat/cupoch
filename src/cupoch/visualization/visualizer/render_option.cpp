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
#include "cupoch/visualization/visualizer/render_option.h"

#include <GL/glew.h>
#include <json/json.h>

#include "cupoch/utility/console.h"

namespace cupoch {
namespace visualization {

bool RenderOption::ConvertToJsonValue(Json::Value &value) const {
    value["class_name"] = "RenderOption";
    value["version_major"] = 1;
    value["version_minor"] = 0;

    if (EigenVector3fToJsonArray(background_color_,
                                 value["background_color"]) == false) {
        return false;
    }
    value["interpolation_option"] = (int)interpolation_option_;

    value["light_on"] = light_on_;
    if (EigenVector3fToJsonArray(light_ambient_color_,
                                 value["light_ambient_color"]) == false) {
        return false;
    }
    if (EigenVector3fToJsonArray(light_position_relative_[0],
                                 value["light0_position"]) == false) {
        return false;
    }
    if (EigenVector3fToJsonArray(light_color_[0], value["light0_color"]) ==
        false) {
        return false;
    }
    value["light0_diffuse_power"] = light_diffuse_power_[0];
    value["light0_specular_power"] = light_specular_power_[0];
    value["light0_specular_shininess"] = light_specular_shininess_[0];
    if (EigenVector3fToJsonArray(light_position_relative_[1],
                                 value["light1_position"]) == false) {
        return false;
    }
    if (EigenVector3fToJsonArray(light_color_[1], value["light1_color"]) ==
        false) {
        return false;
    }
    value["light1_diffuse_power"] = light_diffuse_power_[1];
    value["light1_specular_power"] = light_specular_power_[1];
    value["light1_specular_shininess"] = light_specular_shininess_[1];
    if (EigenVector3fToJsonArray(light_position_relative_[2],
                                 value["light2_position"]) == false) {
        return false;
    }
    if (EigenVector3fToJsonArray(light_color_[2], value["light2_color"]) ==
        false) {
        return false;
    }
    value["light2_diffuse_power"] = light_diffuse_power_[2];
    value["light2_specular_power"] = light_specular_power_[2];
    value["light2_specular_shininess"] = light_specular_shininess_[2];
    if (EigenVector3fToJsonArray(light_position_relative_[3],
                                 value["light3_position"]) == false) {
        return false;
    }
    if (EigenVector3fToJsonArray(light_color_[3], value["light3_color"]) ==
        false) {
        return false;
    }
    value["light3_diffuse_power"] = light_diffuse_power_[3];
    value["light3_specular_power"] = light_specular_power_[3];
    value["light3_specular_shininess"] = light_specular_shininess_[3];

    value["point_size"] = point_size_;
    value["point_color_option"] = (int)point_color_option_;
    value["point_show_normal"] = point_show_normal_;

    value["mesh_shade_option"] = (int)mesh_shade_option_;
    value["mesh_color_option"] = (int)mesh_color_option_;
    value["mesh_show_back_face"] = mesh_show_back_face_;
    value["mesh_show_wireframe"] = mesh_show_wireframe_;
    if (EigenVector3fToJsonArray(default_mesh_color_,
                                 value["default_mesh_color"]) == false) {
        return false;
    }

    value["line_width"] = line_width_;

    value["image_stretch_option"] = (int)image_stretch_option_;
    value["image_max_depth"] = image_max_depth_;

    value["show_coordinate_frame"] = show_coordinate_frame_;
    return true;
}

bool RenderOption::ConvertFromJsonValue(const Json::Value &value) {
    if (value.isObject() == false) {
        utility::LogWarning(
                "ViewTrajectory read JSON failed: unsupported json format.");
        return false;
    }
    if (value.get("class_name", "").asString() != "RenderOption" ||
        value.get("version_major", 1).asInt() != 1 ||
        value.get("version_minor", 0).asInt() != 0) {
        utility::LogWarning(
                "ViewTrajectory read JSON failed: unsupported json format.");
        return false;
    }

    if (EigenVector3fFromJsonArray(background_color_,
                                   value["background_color"]) == false) {
        return false;
    }
    interpolation_option_ =
            (TextureInterpolationOption)value
                    .get("interpolation_option", (int)interpolation_option_)
                    .asInt();

    light_on_ = value.get("light_on", light_on_).asBool();
    if (EigenVector3fFromJsonArray(light_ambient_color_,
                                   value["light_ambient_color"]) == false) {
        return false;
    }
    if (EigenVector3fFromJsonArray(light_position_relative_[0],
                                   value["light0_position"]) == false) {
        return false;
    }
    if (EigenVector3fFromJsonArray(light_color_[0], value["light0_color"]) ==
        false) {
        return false;
    }
    light_diffuse_power_[0] =
            value.get("light0_diffuse_power", light_diffuse_power_[0])
                    .asFloat();
    light_specular_power_[0] =
            value.get("light0_specular_power", light_specular_power_[0])
                    .asFloat();
    light_specular_shininess_[0] =
            value.get("light0_specular_shininess", light_specular_shininess_[0])
                    .asFloat();
    if (EigenVector3fFromJsonArray(light_position_relative_[1],
                                   value["light1_position"]) == false) {
        return false;
    }
    if (EigenVector3fFromJsonArray(light_color_[1], value["light1_color"]) ==
        false) {
        return false;
    }
    light_diffuse_power_[1] =
            value.get("light1_diffuse_power", light_diffuse_power_[1])
                    .asFloat();
    light_specular_power_[1] =
            value.get("light1_specular_power", light_specular_power_[1])
                    .asFloat();
    light_specular_shininess_[1] =
            value.get("light1_specular_shininess", light_specular_shininess_[1])
                    .asFloat();
    if (EigenVector3fFromJsonArray(light_position_relative_[2],
                                   value["light2_position"]) == false) {
        return false;
    }
    if (EigenVector3fFromJsonArray(light_color_[2], value["light2_color"]) ==
        false) {
        return false;
    }
    light_diffuse_power_[2] =
            value.get("light2_diffuse_power", light_diffuse_power_[2])
                    .asFloat();
    light_specular_power_[2] =
            value.get("light2_specular_power", light_specular_power_[2])
                    .asFloat();
    light_specular_shininess_[2] =
            value.get("light2_specular_shininess", light_specular_shininess_[2])
                    .asFloat();
    if (EigenVector3fFromJsonArray(light_position_relative_[3],
                                   value["light3_position"]) == false) {
        return false;
    }
    if (EigenVector3fFromJsonArray(light_color_[3], value["light3_color"]) ==
        false) {
        return false;
    }
    light_diffuse_power_[3] =
            value.get("light3_diffuse_power", light_diffuse_power_[3])
                    .asFloat();
    light_specular_power_[3] =
            value.get("light3_specular_power", light_specular_power_[3])
                    .asFloat();
    light_specular_shininess_[3] =
            value.get("light3_specular_shininess", light_specular_shininess_[3])
                    .asFloat();

    point_size_ = value.get("point_size", point_size_).asFloat();
    point_color_option_ =
            (PointColorOption)value
                    .get("point_color_option", (int)point_color_option_)
                    .asInt();
    point_show_normal_ =
            value.get("point_show_normal", point_show_normal_).asBool();

    mesh_shade_option_ =
            (MeshShadeOption)value
                    .get("mesh_shade_option", (int)mesh_shade_option_)
                    .asInt();
    mesh_color_option_ =
            (MeshColorOption)value
                    .get("mesh_color_option", (int)mesh_color_option_)
                    .asInt();
    mesh_show_back_face_ =
            value.get("mesh_show_back_face", mesh_show_back_face_).asBool();
    mesh_show_wireframe_ =
            value.get("mesh_show_wireframe", mesh_show_wireframe_).asBool();
    if (EigenVector3fFromJsonArray(default_mesh_color_,
                                   value["default_mesh_color"]) == false) {
        return false;
    }

    line_width_ = value.get("line_width", line_width_).asFloat();

    image_stretch_option_ =
            (ImageStretchOption)value
                    .get("image_stretch_option", (int)image_stretch_option_)
                    .asInt();
    image_max_depth_ = value.get("image_max_depth", image_max_depth_).asInt();

    show_coordinate_frame_ =
            value.get("show_coordinate_frame", show_coordinate_frame_).asBool();
    return true;
}

int RenderOption::GetGLDepthFunc() const {
    switch (depthFunc_) {
        case DepthFunc::Never:
            return GL_NEVER;
        case DepthFunc::Less:
            return GL_LESS;
        case DepthFunc::Equal:
            return GL_EQUAL;
        case DepthFunc::LEqual:
            return GL_LEQUAL;
        case DepthFunc::Greater:
            return GL_GREATER;
        case DepthFunc::NotEqual:
            return GL_NOTEQUAL;
        case DepthFunc::GEqual:
            return GL_GEQUAL;
        case DepthFunc::Always:
            return GL_ALWAYS;
    }
    return GL_LESS;  // never hit, makes GCC happy
}

}  // namespace visualization
}  // namespace cupoch