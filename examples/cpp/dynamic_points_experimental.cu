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

    auto included_points_source =
            voxel_source_collisions->CheckIfIncluded(source->points_);
    auto included_points_target =
            voxel_target_collisions->CheckIfIncluded(target->points_);

    cupoch::utility::device_vector<size_t> indic_0, indic_1;
    for (size_t i = 0; i < source->points_.size(); i++) {
        if (included_points_source[i]) {
            indic_0.push_back(i);
        }
    }
    for (size_t i = 0; i < target->points_.size(); i++) {
        if (included_points_target[i]) {
            indic_1.push_back(i);
        }
    }

    auto cld1_source = source->SelectByIndex(indic_0);
    auto cld2_target = target->SelectByIndex(indic_1);

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
            {voxel_target_collisions, voxel_source_collisions, cld1_source,
             cld2_target},
            /*{voxel_target, voxel_source},*/ "Copoch", 640, 480, 50, 50, true,
            true, false);
    return 0;
}
