#include "cupoch/cupoch.h"

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

    // remove the ground
    auto segmented_source = source->SegmentPlane(0.3, 3, 50);
    auto segmented_target = target->SegmentPlane(0.3, 3, 50);
    source = source->SelectByIndex(std::get<1>(segmented_source), true);
    target = target->SelectByIndex(std::get<1>(segmented_target), true);

    // remove noise
    auto denoised_source = source->RemoveStatisticalOutliers(10, 0.2);
    auto denoised_target = target->RemoveStatisticalOutliers(10, 0.2);
    // auto denoised_source = source->RemoveRadiusOutliers(2, 0.2);
    // auto denoised_target = target->RemoveRadiusOutliers(2, 0.2);

    source = std::get<0>(denoised_source);
    target = std::get<0>(denoised_target);

    // estimate normals
    source->EstimateNormals();
    target->EstimateNormals();

    // ICP
    Eigen::Matrix4f eye = Eigen::Matrix4f::Identity();
    auto point_to_point =
            cupoch::registration::TransformationEstimationPointToPoint();
    cupoch::registration::ICPConvergenceCriteria criteria;
    criteria.max_iteration_ = 1000;
    auto res = cupoch::registration::RegistrationICP(*source, *target, 5.0, eye,
                                                     point_to_point, criteria);
    source->Transform(res.transformation_);

    *result = *source + *target;

    // Sampling to mimic keypoints
    auto uniformSampled = result->UniformDownSample(12);
    uniformSampled->PaintUniformColor(Eigen::Vector3f(0, 0, 1));

    // Feature extraction of Keypoints, keypoints mimiced by uniform sampling
    int max_nn = 8;
    float radius = 1.0;
    cupoch::utility::device_vector<Eigen::Vector3f> key_points;
    for (auto&& i : uniformSampled->points_) {
        key_points.push_back(i);
    }
    cupoch::geometry::KDTreeFlann kdtree_source(*source);
    cupoch::geometry::KDTreeFlann kdtree_target(*target);
    cupoch::utility::device_vector<int> indices_source, indices_target;
    cupoch::utility::device_vector<float> distance2_source, distance2_target;

    int result_source = kdtree_source.SearchRadius<Eigen::Vector3f>(
            key_points, radius, max_nn, indices_source, distance2_source);
    int result_target = kdtree_target.SearchRadius<Eigen::Vector3f>(
            key_points, radius, max_nn, indices_target, distance2_target);

    // Lets log some information about keypoints
    cupoch::utility::LogDebug(
            "We have selected {:d} keypoints by uniform sampling",
            key_points.size());

    cupoch::utility::LogDebug(
            "Found source {:d} indices for key point neigbour search",
            indices_source.size());

    cupoch::geometry::KDTreeSearchParamRadius feature_search_param(2 * radius,
                                                                   3 * max_nn);

    // compute fast point feature histograms
    auto fpfh_source = cupoch::registration::ComputeFPFHFeature(*source);
    // auto shot_source = cupoch::registration::ComputeSHOTFeature(*source,
    // 0.8);

    auto fpfh_target = cupoch::registration::ComputeFPFHFeature(*target);
    // auto shot_target = cupoch::registration::ComputeSHOTFeature(*target,
    // 0.8);

    cupoch::utility::LogDebug("Final source cloud has points : {:d}",
                              source->points_.size());
    cupoch::utility::LogDebug("Final target cloud has points : {:d}",
                              target->points_.size());

    for (size_t i = 0; i < indices_source.size() / max_nn; i++) {
      
        cupoch::utility::device_vector<Eigen::Matrix<float, 33, 1>>
                source_keypoint_feature_vector(
                        max_nn, Eigen::Matrix<float, 33, 1>::Zero()),
                target_keypoint_feature_vector(
                        max_nn, Eigen::Matrix<float, 33, 1>::Zero());

        for (size_t j = 0; j < max_nn; j++) {
            int key = i * max_nn + j;

            const int crr_keypoint_nn_source = indices_source[key];
            const int crr_keypoint_nn_target = indices_target[key];

            if (crr_keypoint_nn_source < 0 || crr_keypoint_nn_target < 0) {
                continue;
            }

            thrust::fill(source_keypoint_feature_vector.begin() + j,
                         source_keypoint_feature_vector.begin() + j + 1,
                         fpfh_source->data_[crr_keypoint_nn_source]);

            thrust::fill(target_keypoint_feature_vector.begin() + j,
                         target_keypoint_feature_vector.begin() + j + 1,
                         fpfh_target->data_[crr_keypoint_nn_target]);
        }

        // TODO, use histogram matching to assess similarities of both
        // pointclouds(target, source)
        // TODO, use this similarity value to decide whether there is movement
        // in this region(defined by the keypoint)

        for (size_t f = 0; f < source_keypoint_feature_vector.size(); f++) {
        }
    }

    // Visualize some result and log
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    cupoch::utility::LogDebug("Trial example took : {:f}", ms_double.count());

    cupoch::visualization::DrawGeometries({result, uniformSampled}, "Copoch",
                                          640, 480, 50, 50, true, true, false);
    return 0;
}
