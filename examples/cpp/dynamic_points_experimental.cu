// NOTE, This code is experimental!, not ready to use

#include "cupoch/cupoch.h"

struct saxpy_functor {
    const float a;
    saxpy_functor(float _a) : a(_a) {}
    __device__ float operator()(const float& x, const float& y) const {
        return x / a;
    }
};

struct compare_value {
    __device__ bool operator()(float lhs, float rhs) { return lhs < rhs; }
};

int main(int argc, char* argv[]) {
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;
    auto t1 = high_resolution_clock::now();

    cupoch::utility::InitializeAllocator();
    cupoch::utility::SetVerbosityLevel(cupoch::utility::VerbosityLevel::Debug);
    if (argc < 3) {
        cupoch::utility::LogInfo(
                "Need two arguments of point cloud file name.");
        return 0;
    }

    auto source = std::make_shared<cupoch::geometry::PointCloud>();
    auto target = std::make_shared<cupoch::geometry::PointCloud>();
    auto result = std::make_shared<cupoch::geometry::PointCloud>();

    if (cupoch::io::ReadPointCloud(argv[1], *source)) {
        cupoch::utility::LogInfo("Successfully read {}", argv[1]);
    } else {
        cupoch::utility::LogWarning("Failed to read {}", argv[1]);
    }
    if (cupoch::io::ReadPointCloud(argv[2], *target)) {
        cupoch::utility::LogInfo("Successfully read {}", argv[2]);
    } else {
        cupoch::utility::LogWarning("Failed to read {}", argv[2]);
    }

    // ICP
    Eigen::Matrix4f eye = Eigen::Matrix4f::Identity();
    auto point_to_point =
            cupoch::registration::TransformationEstimationPointToPoint();
    cupoch::registration::ICPConvergenceCriteria criteria;
    criteria.max_iteration_ = 1000;
    auto res = cupoch::registration::RegistrationICP(*source, *target, 5.0, eye,
                                                     point_to_point, criteria);
    source->Transform(res.transformation_);

    // REMOVE THE GROUND
    auto segmented_source = source->SegmentPlane(0.3, 3, 50);
    auto segmented_target = target->SegmentPlane(0.3, 3, 50);
    source = source->SelectByIndex(std::get<1>(segmented_source), true);
    target = target->SelectByIndex(std::get<1>(segmented_target), true);

    // REMOVE THE NOISE
    // auto denoised_source = source->RemoveStatisticalOutliers(10, 0.2);
    // auto denoised_target = target->RemoveStatisticalOutliers(10, 0.2);
    // auto denoised_source = source->RemoveRadiusOutliers(2, 0.2);
    // auto denoised_target = target->RemoveRadiusOutliers(2, 0.2);
    // source = std::get<0>(denoised_source);
    // target = std::get<0>(denoised_target);

    cupoch::utility::LogDebug("Pre-processed source cloud has points : {:d}",
                              source->points_.size());
    cupoch::utility::LogDebug("Pre-processed target cloud has points : {:d}",
                              target->points_.size());

    // NORMAL ESTIMATION
    source->EstimateNormals();
    target->EstimateNormals();

    // MERGE THE SEQUENTIAL CLOUD INTO ONE
    *result = *source + *target;

    // EXTRACT FETAURES OF BOTH CLOUDS(SOURCE TARGET)
    int max_nn = 5;
    float radius = 0.6;
    int uniform_downsample_rate = 5;

    // UNIFORM SAMPLE TO MIMIC KEYPOINTS
    auto uniform_sampled_cloud =
            result->UniformDownSample(uniform_downsample_rate);
    cupoch::utility::device_vector<Eigen::Vector3f> key_points(
            uniform_sampled_cloud->points_.size());
    thrust::copy(uniform_sampled_cloud->points_.begin(),
                 uniform_sampled_cloud->points_.end(), key_points.begin());
    cupoch::utility::LogDebug(
            "We have selected {:d} keypoints by uniform sampling",
            key_points.size());

    // GET THE NEIGHBOURS OF KEYPOINTS FROM SOURCE AND TARGET CLOUDS
    cupoch::geometry::KDTreeFlann kdtree_source(*source);
    cupoch::geometry::KDTreeFlann kdtree_target(*target);
    cupoch::utility::device_vector<int> indices_source, indices_target;
    cupoch::utility::device_vector<float> distance2_source, distance2_target;
    int result_source = kdtree_source.SearchRadius<Eigen::Vector3f>(
            key_points, radius, max_nn, indices_source, distance2_source);
    int result_target = kdtree_target.SearchRadius<Eigen::Vector3f>(
            key_points, radius, max_nn, indices_target, distance2_target);

    cupoch::utility::LogDebug(
            "Found source {:d} indices for key point neigbour search",
            indices_source.size());

    // COMPUTE TARGET AND SOURCE CLOUD FEATURES
    cupoch::geometry::KDTreeSearchParamRadius feature_search_param(radius,
                                                                   max_nn);
    auto fpfh_source = cupoch::registration::ComputeFPFHFeature(
            *source, feature_search_param);
    auto shot_source = cupoch::registration::ComputeSHOTFeature(
            *source, radius, feature_search_param);

    auto fpfh_target = cupoch::registration::ComputeFPFHFeature(
            *target, feature_search_param);
    auto shot_target = cupoch::registration::ComputeSHOTFeature(
            *target, radius, feature_search_param);

    cupoch::utility::device_vector<float> likelihood_of_movement_vector(
            key_points.size(), 0.0);

    // FEATURE MATCHING LOOPS
    float max_likelehood = 0.0;

    // ACTUALLY,
    // indices_source.size() IS EQUAL TO key_points->points_.size() *  max_nn

    for (size_t i = 0; i < indices_source.size() / max_nn; i++) {
        // STORE THE FEATURES OF ALL max_nn KEYPOINT NEIGBOURS FROM SOURCE AND
        // TARGET INTO VECTOR
        cupoch::utility::device_vector<Eigen::Matrix<float, 33, 1>>
                source_keypoint_feature_vector(
                        max_nn, Eigen::Matrix<float, 33, 1>::Zero()),
                target_keypoint_feature_vector(
                        max_nn, Eigen::Matrix<float, 33, 1>::Zero());

        for (size_t j = 0; j < max_nn; j++) {
            // COMPUTE CURRENT KEY TO RETRIVE NEIGBOUR INDICE FROM SOURCE AND
            // TARGET
            int key = i * max_nn + j;
            const int crr_keypoint_nn_source = indices_source[key];
            const int crr_keypoint_nn_target = indices_target[key];

            // IF THE KEYPOINT DOESNT HAVE NEIGBOUR FROM EITHER, CONTINUE
            if (crr_keypoint_nn_source < 0 || crr_keypoint_nn_target < 0) {
                continue;
            }

            thrust::fill(source_keypoint_feature_vector.begin() + j,
                         source_keypoint_feature_vector.begin() + j + 1,
                         &fpfh_source->data_[crr_keypoint_nn_source]);

            thrust::fill(target_keypoint_feature_vector.begin() + j,
                         target_keypoint_feature_vector.begin() + j + 1,
                         &fpfh_target->data_[crr_keypoint_nn_target]);
        }

        // TODO, use histogram matching to assess similarities of both
        // pointclouds(target, source)
        // TODO, use this similarity value to decide whether there is
        // movement in this region(defined by the keypoint)
    }

    /*cupoch::utility::device_vector<float> likelihood_of_movement_vector_norm(
            likelihood_of_movement_vector.size());

    float ted = curr_max_likelehood;

    thrust::transform(likelihood_of_movement_vector.begin(),
                      likelihood_of_movement_vector.end(),
                      likelihood_of_movement_vector_norm.begin(),
                      likelihood_of_movement_vector_norm.begin(),
                      saxpy_functor(curr_max_likelehood));

    cupoch::utility::device_vector<Eigen::Vector3f>
            likelihood_of_movement_vector_colors(
                    likelihood_of_movement_vector.size());

    for (size_t i = 0; i < likelihood_of_movement_vector_norm.size(); i++) {
        thrust::fill(
                likelihood_of_movement_vector_colors.begin() + i,
                likelihood_of_movement_vector_colors.begin() + i + 1,
                Eigen::Vector3f(
                        0, 0,
                        (likelihood_of_movement_vector_norm[i] > 0.4) ? 1 : 0));

        std::cout << likelihood_of_movement_vector_norm[i] << "\n \n";
    }

    std::cout << ted << " tes \n \n";

    uniform_sampled_cloud->SetColors(likelihood_of_movement_vector_colors);*/

    // Visualize some result and log
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    cupoch::utility::LogDebug("Trial example took : {:f}", ms_double.count());

    cupoch::visualization::DrawGeometries({result, uniform_sampled_cloud},
                                          "Copoch", 640, 480, 50, 50, true,
                                          true, false);
    return 0;
}
