#include "cupoch_pybind/registration/registration.h"
#include "cupoch/registration/registration.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/console.h"

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
            const registration::CorrespondenceSet &corres) const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Matrix4f, TransformationEstimationBase,
                               source, target, corres);
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
    te.def("compute_rmse", &registration::TransformationEstimation::ComputeRMSE,
           "source"_a, "target"_a, "corres"_a,
           "Compute RMSE between source and target points cloud given "
           "correspondences.");
    te.def("compute_transformation",
           &registration::TransformationEstimation::ComputeTransformation,
           "source"_a, "target"_a, "corres"_a,
           "Compute transformation from source to target point cloud given "
           "correspondences.");

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
                   return new registration::
                           TransformationEstimationPointToPoint();
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
}

// Registration functions have similar arguments, sharing arg docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"checkers", "checkers"},
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
                 "(``registration::TransformationEstimationPointToPoint``, "
                 "``registration::TransformationEstimationPointToPlane``)"},
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
}

void pybind_registration(py::module &m) {
    py::module m_submodule = m.def_submodule("registration");
    pybind_registration_classes(m_submodule);
    pybind_registration_methods(m_submodule);
}