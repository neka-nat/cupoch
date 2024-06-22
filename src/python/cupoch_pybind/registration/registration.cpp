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
#include "cupoch_pybind/registration/registration.h"

#include "cupoch/geometry/pointcloud.h"
#include "cupoch/registration/colored_icp.h"
#include "cupoch/registration/correspondence_checker.h"
#include "cupoch/registration/fast_global_registration.h"
#include "cupoch/registration/filterreg.h"
#include "cupoch/registration/generalized_icp.h"
#include "cupoch/registration/registration.h"
#include "cupoch/utility/console.h"
#include "cupoch_pybind/docstring.h"

using namespace cupoch;

template <class TransformationEstimationBase =
                  registration::TransformationEstimation>
class PyTransformationEstimation : public TransformationEstimationBase {
public:
    using TransformationEstimationBase::TransformationEstimationBase;
    registration::TransformationEstimationType GetTransformationEstimationType()
            const override {
        PYBIND11_OVERLOAD_PURE(registration::TransformationEstimationType,
                               TransformationEstimationBase, void);
    }
#if !defined(_WIN32)
    float ComputeRMSE(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const registration::CorrespondenceSet &corres) const override {
        PYBIND11_OVERLOAD_PURE(float, TransformationEstimationBase, source,
                               target, corres);
    }
    Eigen::Matrix4f ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const registration::CorrespondenceSet &corres) const {
        PYBIND11_OVERLOAD_PURE(Eigen::Matrix4f, TransformationEstimationBase,
                               source, target, corres);
    }
#endif
};

template <class CorrespondenceCheckerBase = registration::CorrespondenceChecker>
class PyCorrespondenceChecker : public CorrespondenceCheckerBase {
public:
    using CorrespondenceCheckerBase::CorrespondenceCheckerBase;
    bool Check(const geometry::PointCloud &source,
               const geometry::PointCloud &target,
               const registration::CorrespondenceSet &corres,
               const Eigen::Matrix4f &transformation) const override {
        PYBIND11_OVERLOAD_PURE(bool, CorrespondenceCheckerBase, source, target,
                               corres, transformation);
    }
};

void pybind_registration_classes(py::module &m) {
    // cupoch.registration.ICPConvergenceCriteria
    py::class_<registration::ICPConvergenceCriteria> convergence_criteria(
            m, "ICPConvergenceCriteria",
            "Class that defines the convergence criteria of ICP. ICP algorithm "
            "stops if the relative change of fitness and rmse hit "
            "``relative_fitness`` and ``relative_rmse`` individually, or the "
            "iteration number exceeds ``max_iteration``.");
    py::detail::bind_copy_functions<registration::ICPConvergenceCriteria>(
            convergence_criteria);
    convergence_criteria
            .def(py::init([](float fitness, float rmse, int itr) {
                     return new registration::ICPConvergenceCriteria(fitness,
                                                                     rmse, itr);
                 }),
                 "relative_fitness"_a = 1e-6, "relative_rmse"_a = 1e-6,
                 "max_iteration"_a = 30)
            .def_readwrite(
                    "relative_fitness",
                    &registration::ICPConvergenceCriteria::relative_fitness_,
                    "If relative change (difference) of fitness score is lower "
                    "than ``relative_fitness``, the iteration stops.")
            .def_readwrite(
                    "relative_rmse",
                    &registration::ICPConvergenceCriteria::relative_rmse_,
                    "If relative change (difference) of inliner RMSE score is "
                    "lower than ``relative_rmse``, the iteration stops.")
            .def_readwrite(
                    "max_iteration",
                    &registration::ICPConvergenceCriteria::max_iteration_,
                    "Maximum iteration before iteration stops.")
            .def("__repr__", [](const registration::ICPConvergenceCriteria &c) {
                return fmt::format(
                        "registration::ICPConvergenceCriteria class "
                        "with relative_fitness={:e}, relative_rmse={:e}, "
                        "and max_iteration={:d}",
                        c.relative_fitness_, c.relative_rmse_,
                        c.max_iteration_);
            });

    // cupoch.registration.TransformationEstimation
    py::class_<
            registration::TransformationEstimation,
            PyTransformationEstimation<registration::TransformationEstimation>>
            te(m, "TransformationEstimation",
               "Base class that estimates a transformation between two point "
               "clouds. The virtual function ComputeTransformation() must be "
               "implemented in subclasses.");
    te.def("get_transformation_estimation_type", &registration::TransformationEstimation::GetTransformationEstimationType);
    te.def("compute_rmse", &registration::TransformationEstimation::ComputeRMSE,
           "source"_a, "target"_a, "corres"_a,
           "Compute RMSE between source and target points cloud given "
           "correspondences.");
    te.def("compute_transformation",
           py::overload_cast<const geometry::PointCloud&, const geometry::PointCloud&, const registration::CorrespondenceSet&>(
                &registration::TransformationEstimation::ComputeTransformation, py::const_),
           "source"_a, "target"_a, "corres"_a,
           "Compute transformation from source to target point cloud given "
           "correspondences.");

    py::enum_<registration::TransformationEstimationType> tf_type(m, "TransformationEstimationType");
    tf_type.value("PointToPoint", registration::TransformationEstimationType::PointToPoint)
            .value("PointToPlane", registration::TransformationEstimationType::PointToPlane)
            .value("SymmetricMethod", registration::TransformationEstimationType::SymmetricMethod)
            .value("ColoredICP", registration::TransformationEstimationType::ColoredICP)
            .value("GeneralizedICP", registration::TransformationEstimationType::GeneralizedICP)
            .export_values();

    // cupoch.registration.TransformationEstimationPointToPoint:
    // TransformationEstimation
    py::class_<registration::TransformationEstimationPointToPoint,
               PyTransformationEstimation<
                       registration::TransformationEstimationPointToPoint>,
               registration::TransformationEstimation>
            te_p2p(m, "TransformationEstimationPointToPoint",
                   "Class to estimate a transformation for point to point "
                   "distance.");
    py::detail::bind_copy_functions<
            registration::TransformationEstimationPointToPoint>(te_p2p);
    te_p2p.def(py::init([]() {
              return new registration::TransformationEstimationPointToPoint();
          }))
            .def("__repr__",
                 [](const registration::TransformationEstimationPointToPoint
                            &te) {
                     return std::string(
                             "registration::"
                             "TransformationEstimationPointToPoint ");
                 });

    // cupoch.registration.TransformationEstimationPointToPlane:
    // TransformationEstimation
    py::class_<registration::TransformationEstimationPointToPlane,
               PyTransformationEstimation<
                       registration::TransformationEstimationPointToPlane>,
               registration::TransformationEstimation>
            te_p2l(m, "TransformationEstimationPointToPlane",
                   "Class to estimate a transformation for point to plane "
                   "distance.");
    py::detail::bind_default_constructor<
            registration::TransformationEstimationPointToPlane>(te_p2l);
    py::detail::bind_copy_functions<
            registration::TransformationEstimationPointToPlane>(te_p2l);
    te_p2l.def(
            "__repr__",
            [](const registration::TransformationEstimationPointToPlane &te) {
                return std::string("TransformationEstimationPointToPlane");
            });

    // cupoch.registration.TransformationEstimationSymmetricMethod:
    // TransformationEstimation
    py::class_<registration::TransformationEstimationSymmetricMethod,
               PyTransformationEstimation<
                       registration::TransformationEstimationSymmetricMethod>,
               registration::TransformationEstimation>
            te_l2l(m, "TransformationEstimationSymmetricMethod",
                   "Class to estimate a transformation for plane to plane "
                   "distance.");
    py::detail::bind_default_constructor<
            registration::TransformationEstimationSymmetricMethod>(te_l2l);
    py::detail::bind_copy_functions<
            registration::TransformationEstimationSymmetricMethod>(te_l2l);
    te_l2l.def(
            "__repr__",
            [](const registration::TransformationEstimationSymmetricMethod &te) {
                return std::string("TransformationEstimationSymmetricMethod");
            });

    // cupoch.registration.TransformationEstimationForGeneralizedICP:
    // TransformationEstimation
    py::class_<registration::TransformationEstimationForGeneralizedICP,
               PyTransformationEstimation<
                       registration::TransformationEstimationForGeneralizedICP>,
               registration::TransformationEstimation>
            te_gicp(m, "TransformationEstimationForGeneralizedICP",
                    "Class to estimate a transformation for Generalized ICP.");
    py::detail::bind_copy_functions<
            registration::TransformationEstimationForGeneralizedICP>(te_gicp);
    te_gicp.def(py::init([](float epsilon) {
                  return new registration::TransformationEstimationForGeneralizedICP(
                          epsilon);
              }),
              "epsilon"_a = 1e-3)
            .def_readwrite("epsilon",
                           &registration::TransformationEstimationForGeneralizedICP::
                                   epsilon_,
                           "Small constant representing covariance along the "
                           "normal.")
            .def("__repr__",
                 [](const registration::TransformationEstimationForGeneralizedICP
                            &te) {
                     return std::string(
                             "registration::"
                             "TransformationEstimationForGeneralizedICP");
                 });

    // cupoch.registration.CorrespondenceChecker
    py::class_<registration::CorrespondenceChecker,
               PyCorrespondenceChecker<registration::CorrespondenceChecker>>
            cc(m, "CorrespondenceChecker",
               "Base class that checks if two (small) point clouds can be "
               "aligned. This class is used in feature based matching "
               "algorithms (such as RANSAC and FastGlobalRegistration) to "
               "prune out outlier correspondences. The virtual function "
               "Check() must be implemented in subclasses.");
    cc.def("Check", &registration::CorrespondenceChecker::Check, "source"_a, "target"_a,
           "corres"_a, "transformation"_a,
           "Function to check if two points can be aligned. The two input "
           "point clouds must have exact the same number of points.");
    cc.def_readwrite(
            "require_pointcloud_alignment_",
            &registration::CorrespondenceChecker::require_pointcloud_alignment_,
            "Some checkers do not require point clouds to be aligned, e.g., "
            "the edge length checker. Some checkers do, e.g., the distance "
            "checker.");
    docstring::ClassMethodDocInject(
            m, "CorrespondenceChecker", "Check",
            {{"source", "Source point cloud."},
             {"target", "Target point cloud."},
             {"corres",
              "Correspondence set between source and target point cloud."},
             {"transformation", "The estimated transformation (inplace)."}});

    // cupoch.registration.CorrespondenceCheckerBasedOnEdgeLength:
    // CorrespondenceChecker
    py::class_<registration::CorrespondenceCheckerBasedOnEdgeLength,
               PyCorrespondenceChecker<registration::CorrespondenceCheckerBasedOnEdgeLength>,
               registration::CorrespondenceChecker>
            cc_el(m, "CorrespondenceCheckerBasedOnEdgeLength",
                  "Check if two point clouds build the polygons with similar "
                  "edge lengths. That is, checks if the lengths of any two "
                  "arbitrary edges (line formed by two vertices) individually "
                  "drawn withinin source point cloud and within the target "
                  "point cloud with correspondences are similar. The only "
                  "parameter similarity_threshold is a number between 0 "
                  "(loose) and 1 (strict)");
    py::detail::bind_copy_functions<registration::CorrespondenceCheckerBasedOnEdgeLength>(
            cc_el);
    cc_el.def(py::init([](float similarity_threshold) {
                  return new registration::CorrespondenceCheckerBasedOnEdgeLength(
                          similarity_threshold);
              }),
              "similarity_threshold"_a = 0.9)
            .def("__repr__",
                 [](const registration::CorrespondenceCheckerBasedOnEdgeLength &c) {
                     return fmt::format(
                             ""
                             "CorrespondenceCheckerBasedOnEdgeLength "
                             "with similarity_threshold={:f}",
                             c.similarity_threshold_);
                 })
            .def_readwrite(
                    "similarity_threshold",
                    &registration::CorrespondenceCheckerBasedOnEdgeLength::
                            similarity_threshold_,
                    R"(float value between 0 (loose) and 1 (strict): For the
check to be true,

:math:`||\text{edge}_{\text{source}}|| > \text{similarity_threshold} \times ||\text{edge}_{\text{target}}||` and

:math:`||\text{edge}_{\text{target}}|| > \text{similarity_threshold} \times ||\text{edge}_{\text{source}}||`

must hold true for all edges.)");

    // cupoch.registration.CorrespondenceCheckerBasedOnDistance:
    // CorrespondenceChecker
    py::class_<registration::CorrespondenceCheckerBasedOnDistance,
               PyCorrespondenceChecker<registration::CorrespondenceCheckerBasedOnDistance>,
               registration::CorrespondenceChecker>
            cc_d(m, "CorrespondenceCheckerBasedOnDistance",
                 "Class to check if aligned point clouds are close (less than "
                 "specified threshold).");
    py::detail::bind_copy_functions<registration::CorrespondenceCheckerBasedOnDistance>(cc_d);
    cc_d.def(py::init([](float distance_threshold) {
                 return new registration::CorrespondenceCheckerBasedOnDistance(
                         distance_threshold);
             }),
             "distance_threshold"_a)
            .def("__repr__",
                 [](const registration::CorrespondenceCheckerBasedOnDistance &c) {
                     return fmt::format(
                             ""
                             "CorrespondenceCheckerBasedOnDistance with "
                             "distance_threshold={:f}",
                             c.distance_threshold_);
                 })
            .def_readwrite(
                    "distance_threshold",
                    &registration::CorrespondenceCheckerBasedOnDistance::distance_threshold_,
                    "Distance threshold for the check.");

    // cupoch.registration.CorrespondenceCheckerBasedOnNormal:
    // CorrespondenceChecker
    py::class_<registration::CorrespondenceCheckerBasedOnNormal,
               PyCorrespondenceChecker<registration::CorrespondenceCheckerBasedOnNormal>,
               registration::CorrespondenceChecker>
            cc_n(m, "CorrespondenceCheckerBasedOnNormal",
                 "Class to check if two aligned point clouds have similar "
                 "normals. It considers vertex normal affinity of any "
                 "correspondences. It computes dot product of two normal "
                 "vectors. It takes radian value for the threshold.");
    py::detail::bind_copy_functions<registration::CorrespondenceCheckerBasedOnNormal>(cc_n);
    cc_n.def(py::init([](float normal_angle_threshold) {
                 return new registration::CorrespondenceCheckerBasedOnNormal(
                         normal_angle_threshold);
             }),
             "normal_angle_threshold"_a)
            .def("__repr__",
                 [](const registration::CorrespondenceCheckerBasedOnNormal &c) {
                     return fmt::format(
                             ""
                             "CorrespondenceCheckerBasedOnNormal with "
                             "normal_threshold={:f}",
                             c.normal_angle_threshold_);
                 })
            .def_readwrite("normal_angle_threshold",
                           &registration::CorrespondenceCheckerBasedOnNormal::
                                   normal_angle_threshold_,
                           "Radian value for angle threshold.");

    // cupoch.registration.FastGlobalRegistrationOption:
    py::class_<registration::FastGlobalRegistrationOption> fgr_option(
            m, "FastGlobalRegistrationOption",
            "Options for FastGlobalRegistration.");
    py::detail::bind_copy_functions<registration::FastGlobalRegistrationOption>(
            fgr_option);
    fgr_option
            .def(py::init([](float division_factor, bool use_absolute_scale,
                             bool decrease_mu,
                             float maximum_correspondence_distance,
                             int iteration_number, float tuple_scale,
                             int maximum_tuple_count) {
                     return new registration::FastGlobalRegistrationOption(
                             division_factor, use_absolute_scale, decrease_mu,
                             maximum_correspondence_distance, iteration_number,
                             tuple_scale, maximum_tuple_count);
                 }),
                 "division_factor"_a = 1.4, "use_absolute_scale"_a = false,
                 "decrease_mu"_a = false,
                 "maximum_correspondence_distance"_a = 0.025,
                 "iteration_number"_a = 64, "tuple_scale"_a = 0.95,
                 "maximum_tuple_count"_a = 1000)
            .def_readwrite(
                    "division_factor",
                    &registration::FastGlobalRegistrationOption::
                            division_factor_,
                    "float: Division factor used for graduated non-convexity.")
            .def_readwrite(
                    "use_absolute_scale",
                    &registration::FastGlobalRegistrationOption::
                            use_absolute_scale_,
                    "bool: Measure distance in absolute scale (1) or in scale "
                    "relative to the diameter of the model (0).")
            .def_readwrite(
                    "decrease_mu",
                    &registration::FastGlobalRegistrationOption::decrease_mu_,
                    "bool: Set to ``True`` to decrease scale mu by "
                    "``division_factor`` for graduated non-convexity.")
            .def_readwrite("maximum_correspondence_distance",
                           &registration::FastGlobalRegistrationOption::
                                   maximum_correspondence_distance_,
                           "float: Maximum correspondence distance.")
            .def_readwrite("iteration_number",
                           &registration::FastGlobalRegistrationOption::
                                   iteration_number_,
                           "int: Maximum number of iterations.")
            .def_readwrite(
                    "tuple_scale",
                    &registration::FastGlobalRegistrationOption::tuple_scale_,
                    "float: Similarity measure used for tuples of feature "
                    "points.")
            .def_readwrite("maximum_tuple_count",
                           &registration::FastGlobalRegistrationOption::
                                   maximum_tuple_count_,
                           "float: Maximum tuple numbers.")
            .def("__repr__",
                 [](const registration::FastGlobalRegistrationOption &c) {
                     return fmt::format(
                             "registration::"
                             "FastGlobalRegistrationOption class "
                             "with \ndivision_factor={}"
                             "\nuse_absolute_scale={}"
                             "\ndecrease_mu={}"
                             "\nmaximum_correspondence_distance={}"
                             "\niteration_number={}"
                             "\ntuple_scale={}"
                             "\nmaximum_tuple_count={}",
                             c.division_factor_, c.use_absolute_scale_,
                             c.decrease_mu_, c.maximum_correspondence_distance_,
                             c.iteration_number_, c.tuple_scale_,
                             c.maximum_tuple_count_);
                 });

    // cupoch.registration.FilterRegOption:
    py::class_<registration::FilterRegOption> filterreg_option(
            m, "FilterRegOption", "Options for FilterReg.");
    py::detail::bind_copy_functions<registration::FilterRegOption>(
            filterreg_option);
    filterreg_option
            .def(py::init([](float sigma_initial, float sigma_min,
                             float relative_likelihood, int max_iteration) {
                     return new registration::FilterRegOption(
                             sigma_initial, sigma_min, relative_likelihood,
                             max_iteration);
                 }),
                 "sigma_initial"_a = 0.1, "sigma_min"_a = 1e-4,
                 "relative_likelihood"_a = 1.0e-6, "max_iteration"_a = 20)
            .def_readwrite("sigma_initial",
                           &registration::FilterRegOption::sigma_initial_,
                           "float: Initial value of the variance of the "
                           "Gaussian distribution.")
            .def_readwrite("sigma_min",
                           &registration::FilterRegOption::sigma_min_,
                           "float: Minimum value of the variance of the "
                           "Gaussian distribution.")
            .def_readwrite("max_iteration",
                           &registration::FilterRegOption::max_iteration_,
                           "int: Maximum number of iterations.")
            .def("__repr__", [](const registration::FilterRegOption &c) {
                return fmt::format(
                        "registration::"
                        "FilterRegOption class "
                        "with \nsigma_initial={}"
                        "\nsigma_min={}"
                        "\nmax_iteration={}",
                        c.sigma_initial_, c.sigma_min_, c.max_iteration_);
            });

    // cupoch.registration.RegistrationResult
    py::class_<registration::RegistrationResult> registration_result(
            m, "RegistrationResult",
            "Class that contains the registration results.");
    py::detail::bind_default_constructor<registration::RegistrationResult>(
            registration_result);
    py::detail::bind_copy_functions<registration::RegistrationResult>(
            registration_result);
    registration_result
            .def_readwrite("transformation",
                           &registration::RegistrationResult::transformation_,
                           "``4 x 4`` float32 numpy array: The estimated "
                           "transformation matrix.")
            .def_property(
                    "correspondence_set",
                    &registration::RegistrationResult::GetCorrespondenceSet,
                    &registration::RegistrationResult::SetCorrespondenceSet,
                    "``n x 2`` int numpy array: Correspondence set between "
                    "source and target point cloud.")
            .def_readwrite("inlier_rmse",
                           &registration::RegistrationResult::inlier_rmse_,
                           "float: RMSE of all inlier correspondences. Lower "
                           "is better.")
            .def_readwrite(
                    "fitness", &registration::RegistrationResult::fitness_,
                    "float: The overlapping area (# of inlier correspondences "
                    "/ # of points in target). Higher is better.")
            .def("__repr__", [](const registration::RegistrationResult &rr) {
                return std::string(
                               "registration::RegistrationResult with fitness "
                               "= ") +
                       std::to_string(rr.fitness_) +
                       std::string(", inlier_rmse = ") +
                       std::to_string(rr.inlier_rmse_) +
                       std::string(", and correspondence_set size of ") +
                       std::to_string(rr.correspondence_set_.size()) +
                       std::string("\nAccess transformation to get result.");
            });

    // cupoch.registration.FilterRegResult
    py::class_<registration::FilterRegResult> filterreg_result(
            m, "FilterRegResult",
            "Class that contains the FilterReg registration results.");
    py::detail::bind_default_constructor<registration::FilterRegResult>(
            filterreg_result);
    py::detail::bind_copy_functions<registration::FilterRegResult>(
            filterreg_result);
    filterreg_result
            .def_readwrite("transformation",
                           &registration::FilterRegResult::transformation_,
                           "``4 x 4`` float32 numpy array: The estimated "
                           "transformation matrix.")
            .def_readwrite("likelihood",
                           &registration::FilterRegResult::likelihood_,
                           "float: The likelihood (# of inlier correspondences "
                           "/ # of points in target). Higher is better.")
            .def("__repr__", [](const registration::FilterRegResult &rr) {
                return std::string(
                               "registration::FilterRegResult with likelihood "
                               "= ") +
                       std::to_string(rr.likelihood_) +
                       std::string("\nAccess transformation to get result.");
            });
}

// Registration functions have similar arguments, sharing arg docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"checkers",
                 "Vector of Checker class to check if two point "
                 "clouds can be aligned. One of "
                 "(``CorrespondenceCheckerBasedOnEdgeLength``, "
                 "``CorrespondenceCheckerBasedOnDistance``, "
                 "``CorrespondenceCheckerBasedOnNormal``)"},
                {"corres",
                 "Checker class to check if two point clouds can be "
                 "aligned. "
                 "One of "
                 "(``registration::CorrespondenceCheckerBasedOnEdgeLength``, "
                 "``registration::CorrespondenceCheckerBasedOnDistance``, "
                 "``registration::CorrespondenceCheckerBasedOnNormal``)"},
                {"criteria", "Convergence criteria"},
                {"estimation_method",
                 "Estimation method. One of "
                 "(``registration::TransformationEstimationPointToPoint``,"
                 "``registration::TransformationEstimationPointToPlane``,"
                 "``registration::TransformationEstimationSymmetricMethod``,"
                 "``registration::TransformationEstimationForGeneralizedICP``)"},
                {"init", "Initial transformation estimation"},
                {"lambda_geometric", "lambda_geometric value"},
                {"max_correspondence_distance",
                 "Maximum correspondence points-pair distance."},
                {"option", "Registration option"},
                {"ransac_n", "Fit ransac with ``ransac_n`` correspondences"},
                {"source_feature", "Source point cloud feature."},
                {"source", "The source point cloud."},
                {"target_feature", "Target point cloud feature."},
                {"target", "The target point cloud."},
                {"transformation",
                 "The 4x4 transformation matrix to transform ``source`` to "
                 "``target``"}};

void pybind_registration_methods(py::module &m) {
    m.def("registration_icp", &registration::RegistrationICP,
          "Function for ICP registration", "source"_a, "target"_a,
          "max_correspondence_distance"_a,
          "init"_a = Eigen::Matrix4f::Identity(),
          "estimation_method"_a =
                  registration::TransformationEstimationPointToPoint(),
          "criteria"_a = registration::ICPConvergenceCriteria());
    docstring::FunctionDocInject(m, "registration_icp",
                                 map_shared_argument_docstrings);

    m.def("registration_colored_icp", &registration::RegistrationColoredICP,
          "Function for Colored ICP registration", "source"_a, "target"_a,
          "max_correspondence_distance"_a,
          "init"_a = Eigen::Matrix4f::Identity(),
          "criteria"_a = registration::ICPConvergenceCriteria(),
          "lambda_geometric"_a = 0.968,
          "det_thresh"_a = 1.0e-6);
    docstring::FunctionDocInject(m, "registration_colored_icp",
                                 map_shared_argument_docstrings);

    m.def("registration_generalized_icp", &registration::RegistrationGeneralizedICP,
          "Function for Generalized ICP registration", "source"_a, "target"_a,
          "max_correspondence_distance"_a,
          "init"_a = Eigen::Matrix4f::Identity(),
          "estimation"_a = registration::TransformationEstimationForGeneralizedICP(),
          "criteria"_a = registration::ICPConvergenceCriteria());

    m.def("registration_ransac_based_on_correspondence",
          &registration::RegistrationRANSACBasedOnCorrespondence,
          py::call_guard<py::gil_scoped_release>(),
          "Function for global RANSAC registration based on a set of "
          "correspondences",
          "source"_a, "target"_a, "corres"_a, "max_correspondence_distance"_a,
          "estimation_method"_a = registration::TransformationEstimationPointToPoint(),
          "ransac_n"_a = 3,
          "checkers"_a = std::vector<
                  std::reference_wrapper<const registration::CorrespondenceChecker>>(),
          "criteria"_a = registration::RANSACConvergenceCriteria(100000, 0.999));
    docstring::FunctionDocInject(m,
                                 "registration_ransac_based_on_correspondence",
                                 map_shared_argument_docstrings);

    m.def("registration_fast_based_on_feature_matching",
          &registration::FastGlobalRegistration<33>,
          "Function for fast global registration based on feature matching",
          "source"_a, "target"_a, "source_feature"_a, "target_feature"_a,
          "option"_a = registration::FastGlobalRegistrationOption());
    m.def("registration_fast_based_on_feature_matching",
          &registration::FastGlobalRegistration<352>,
          "Function for fast global registration based on feature matching",
          "source"_a, "target"_a, "source_feature"_a, "target_feature"_a,
          "option"_a = registration::FastGlobalRegistrationOption());
    docstring::FunctionDocInject(m,
                                 "registration_fast_based_on_feature_matching",
                                 map_shared_argument_docstrings);

    m.def("registration_filterreg", &registration::RegistrationFilterReg,
          "Function for FilterReg", "source"_a, "target"_a,
          "init"_a = Eigen::Matrix4f::Identity(),
          "criteria"_a = registration::FilterRegOption());
    docstring::FunctionDocInject(m, "registration_filterreg",
                                 map_shared_argument_docstrings);
}

void pybind_registration(py::module &m) {
    py::module m_submodule = m.def_submodule("registration");
    pybind_registration_classes(m_submodule);
    pybind_registration_methods(m_submodule);
    pybind_feature(m_submodule);
    pybind_feature_methods(m_submodule);
}