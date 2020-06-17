#pragma once

#include <Eigen/Core>
#include "cupoch/registration/feature.h"

namespace cupoch {

namespace geometry {
class PointCloud;
}

namespace registration {

class RegistrationResult;

/// \class FastGlobalRegistrationOption
///
/// \brief Options for FastGlobalRegistration.
class FastGlobalRegistrationOption {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param division_factor Division factor used for graduated non-convexity.
    /// \param use_absolute_scale Measure distance in absolute scale (1) or in
    /// scale relative to the diameter of the model (0).
    /// \param decrease_mu Set
    /// to `true` to decrease scale mu by division_factor for graduated
    /// non-convexity.
    /// \param maximum_correspondence_distance Maximum
    /// correspondence distance (also see comment of USE_ABSOLUTE_SCALE).
    /// \param iteration_number Maximum number of iterations.
    /// \param tuple_scale Similarity measure used for tuples of feature points.
    /// \param maximum_tuple_count Maximum numer of tuples.
    FastGlobalRegistrationOption(float division_factor = 1.4,
                                 bool use_absolute_scale = false,
                                 bool decrease_mu = true,
                                 float maximum_correspondence_distance = 0.025,
                                 int iteration_number = 64,
                                 float tuple_scale = 0.95,
                                 int maximum_tuple_count = 1000)
        : division_factor_(division_factor),
          use_absolute_scale_(use_absolute_scale),
          decrease_mu_(decrease_mu),
          maximum_correspondence_distance_(maximum_correspondence_distance),
          iteration_number_(iteration_number),
          tuple_scale_(tuple_scale),
          maximum_tuple_count_(maximum_tuple_count) {}
    ~FastGlobalRegistrationOption() {}

public:
    /// Division factor used for graduated non-convexity.
    float division_factor_;
    /// Measure distance in absolute scale (1) or in scale relative to the
    /// diameter of the model (0).
    bool use_absolute_scale_;
    /// Set to `true` to decrease scale mu by division_factor for graduated
    /// non-convexity.
    bool decrease_mu_;
    /// Maximum correspondence distance (also see comment of
    /// USE_ABSOLUTE_SCALE).
    float maximum_correspondence_distance_;
    /// Maximum number of iterations.
    int iteration_number_;
    /// Similarity measure used for tuples of feature points.
    float tuple_scale_;
    /// Maximum number of tuples..
    int maximum_tuple_count_;
};

template <int Dim>
RegistrationResult FastGlobalRegistration(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const Feature<Dim> &source_feature,
        const Feature<Dim> &target_feature,
        const FastGlobalRegistrationOption &option =
                FastGlobalRegistrationOption());

}  // namespace registration
}  // namespace cupoch