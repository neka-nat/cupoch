// NOTE, This code is experimental!, not ready to use
#include "cupoch/cupoch.h"

Eigen::Vector3f getColorByIndex(int index) {
    Eigen::Vector3f result;
    switch (index) {
        case 0:  // RED:
            result[0] = 0.8;
            result[1] = 0.1;
            result[2] = 0.1;
            break;
        case 1:  // GREEN:
            result[0] = 0.1;
            result[1] = 0.8;
            result[2] = 0.1;
            break;
        case 2:  // GREY:
            result[0] = 0.9;
            result[1] = 0.9;
            result[2] = 0.9;
            break;
        case 3:  // DARK_GREY:
            result[0] = 0.6;
            result[1] = 0.6;
            result[2] = 0.6;
            break;
        case 4:  // WHITE:
            result[0] = 1.0;
            result[1] = 1.0;
            result[2] = 1.0;
            break;
        case 5:  // ORANGE:
            result[0] = 1.0;
            result[1] = 0.5;
            result[2] = 0.0;
            break;
        case 6:  // Maroon:
            result[0] = 0.5;
            result[1] = 0.0;
            result[2] = 0.0;
            break;
        case 7:  // Olive:
            result[0] = 0.5;
            result[1] = 0.5;
            result[2] = 0.0;
            break;
        case 8:  // Navy:
            result[0] = 0.0;
            result[1] = 0.0;
            result[2] = 0.5;
            break;
        case 9:  // BLACK:
            result[0] = 0.0;
            result[1] = 0.0;
            result[2] = 0.0;
            break;
        case 10:  // YELLOW:
            result[0] = 1.0;
            result[1] = 1.0;
            result[2] = 0.0;
            break;
        case 11:  // BROWN:
            result[0] = 0.597;
            result[1] = 0.296;
            result[2] = 0.0;
            break;
        case 12:  // PINK:
            result[0] = 1.0;
            result[1] = 0.4;
            result[2] = 1;
            break;
        case 13:  // LIME_GREEN:
            result[0] = 0.6;
            result[1] = 1.0;
            result[2] = 0.2;
            break;
        case 14:  // PURPLE:
            result[0] = 0.597;
            result[1] = 0.0;
            result[2] = 0.597;
            break;
        case 15:  // CYAN:
            result[0] = 0.0;
            result[1] = 1.0;
            result[2] = 1.0;
            break;
        case 16:  // MAGENTA:
            result[0] = 1.0;
            result[1] = 0.0;
            result[2] = 1.0;
    }
    return result;
}

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
    auto segmented_source = source->SegmentPlane(0.4, 3, 50);
    auto segmented_target = target->SegmentPlane(0.4, 3, 50);
    source = source->SelectByIndex(std::get<1>(segmented_source), true);
    target = target->SelectByIndex(std::get<1>(segmented_target), true);

    // REMOVE THE NOISE
    auto denoised_source = source->RemoveStatisticalOutliers(10, 0.1);
    auto denoised_target = target->RemoveStatisticalOutliers(10, 0.1);
    denoised_source =
            std::get<0>(denoised_source)->RemoveRadiusOutliers(2, 0.2);
    denoised_target =
            std::get<0>(denoised_target)->RemoveRadiusOutliers(2, 0.2);
    source = std::get<0>(denoised_source);
    target = std::get<0>(denoised_target);

    cupoch::utility::LogDebug("Pre-processed source cloud has points : {:d}",
                              source->points_.size());
    cupoch::utility::LogDebug("Pre-processed target cloud has points : {:d}",
                              target->points_.size());

    // NORMAL ESTIMATION
    source->EstimateNormals();
    target->EstimateNormals();
    // MERGE THE SEQUENTIAL CLOUD INTO ONE
    *result = *source + *target;

    // DBS CLUSTERING ?
    auto source_cluster_labels = source->ClusterDBSCAN(0.9, 10, false, 100);
    auto target_cluster_labels = target->ClusterDBSCAN(0.9, 10, false, 100);
    auto result_cluster_labels = result->ClusterDBSCAN(1.0, 10, false, 100);

    // EXTRACT FETAURES OF BOTH CLOUDS(SOURCE TARGET)
    int max_nn = 5;
    float radius = 0.5;
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

    /*for (size_t i = 0; i < indices_source.size() / max_nn; i++) {
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
                         fpfh_source->data_[crr_keypoint_nn_source]);

            thrust::fill(target_keypoint_feature_vector.begin() + j,
                         target_keypoint_feature_vector.begin() + j + 1,
                         fpfh_target->data_[crr_keypoint_nn_target]);
        }

        // TODO, use histogram matching to assess similarities of both
        // pointclouds(target, source)
        // TODO, use this similarity value to decide whether there is
        // movement in this region(defined by the keypoint)
    }*/

    cupoch::registration::FastGlobalRegistrationOption option;
    option.use_absolute_scale_ = true;
    option.maximum_correspondence_distance_ = 0.5;
    option.tuple_scale_ = 0.9;

    auto registration_result = cupoch::registration::FastGlobalRegistration<33>(
            *source, *target, *fpfh_source, *fpfh_target, option);

    thrust::host_vector<Eigen::Vector3f> points(
            registration_result.correspondence_set_.size() * 2,
            Eigen::Vector3f(0, 0, 0));

    auto pairs = registration_result.GetCorrespondenceSet();

    thrust::host_vector<Eigen::Vector2i> new_pairs(pairs.size(),
                                                   Eigen::Vector2i(0, 0));
    for (size_t i = 0; i < registration_result.correspondence_set_.size();
         i++) {
        points[2 * i] = source->points_[pairs[i].x()];
        points[2 * i + 1] = target->points_[pairs[i].y()];
        new_pairs[i] = Eigen::Vector2i(2 * i, 2 * i + 1);
    }
    auto correspondance_mesh =
            std::make_shared<cupoch::geometry::LineSet<3>>(points, new_pairs);

    for (size_t i = 0; i < result->colors_.size(); i++) {
        if (result_cluster_labels[i] < 0) {
            result->colors_[i] = Eigen::Vector3f(0, 0, 0);
        } else {
            result->colors_[i] = getColorByIndex(result_cluster_labels[i] % 16);
        }
    }

    double voxel_size = 0.25;

    auto voxel_source = cupoch::geometry::VoxelGrid::CreateFromPointCloud(
            *source, voxel_size);
    auto voxel_target = cupoch::geometry::VoxelGrid::CreateFromPointCloud(
            *target, voxel_size);

    voxel_source->PaintUniformColor(getColorByIndex(0));
    voxel_target->PaintUniformColor(getColorByIndex(1));

    auto collision_result = cupoch::collision::ComputeIntersection(
            *voxel_target, *voxel_source, 0.0);

    auto target_collision_indices =
            collision_result->GetFirstCollisionIndices();
    auto source_collision_indices =
            collision_result->GetSecondCollisionIndices();

    voxel_source->PaintIndexedColor(source_collision_indices,
                                    getColorByIndex(2));
    voxel_target->PaintIndexedColor(target_collision_indices,
                                    getColorByIndex(2));

    auto voxel_target_collisions =
            std::make_shared<cupoch::geometry::VoxelGrid>();
    voxel_target_collisions->voxel_size_ = voxel_size;
    auto voxel_source_collisions =
            std::make_shared<cupoch::geometry::VoxelGrid>();
    voxel_source_collisions->voxel_size_ = voxel_size;

    voxel_target->SelectByIndexImpl(*voxel_target, *voxel_target_collisions,
                                    target_collision_indices, true);
    voxel_source->SelectByIndexImpl(*voxel_source, *voxel_source_collisions,
                                    source_collision_indices, true);
    
    // Visualize some result and log
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    cupoch::utility::LogDebug("Trial example took : {:f}", ms_double.count());

    cupoch::visualization::DrawGeometries(
            {voxel_target_collisions, voxel_source_collisions},
            /*{voxel_target, voxel_source},*/ "Copoch", 640, 480, 50, 50, true,
            true, false);
    return 0;
}
