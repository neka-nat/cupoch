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

    // compute fast point feature histograms
    auto fpfh_source = cupoch::registration::ComputeFPFHFeature(*source);
    auto shot_source = cupoch::registration::ComputeSHOTFeature(*source, 0.8);

    auto fpfh_target = cupoch::registration::ComputeFPFHFeature(*target);
    auto shot_target = cupoch::registration::ComputeSHOTFeature(*target, 0.8);

    std::cout << fpfh_source->GetData()[0] << std::endl;
    std::cout << shot_source->GetData()[0] << std::endl;
    std::cout << fpfh_target->GetData()[0] << std::endl;
    std::cout << shot_target->GetData()[0] << std::endl;

    // ICP
    Eigen::Matrix4f eye = Eigen::Matrix4f::Identity();
    auto point_to_point =
            cupoch::registration::TransformationEstimationPointToPoint();
    cupoch::registration::ICPConvergenceCriteria criteria;
    criteria.max_iteration_ = 1000;
    auto res = cupoch::registration::RegistrationICP(*source, *target, 5.0, eye,
                                                     point_to_point, criteria);
    source->Transform(res.transformation_);

    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    // Sampling
    auto uniformSampled = result->UniformDownSample(10);
    uniformSampled->PaintUniformColor(Eigen::Vector3f(0, 0, 255));

    // Feature extraction
    int max_nn = 8;
    float radius = 1.0;
    cupoch::utility::device_vector<Eigen::Vector3f> key_points;
    key_points.push_back(Eigen::Vector3f(0, 0, 0));
    key_points.push_back(Eigen::Vector3f(0, 0, 0));

    cupoch::geometry::KDTreeFlann kdtree_source(*source);
    cupoch::geometry::KDTreeFlann kdtree_target(*target);

    cupoch::utility::device_vector<int> indices_source, indices_target;
    cupoch::utility::device_vector<float> distance2_source, distance2_target;

    int result_source = kdtree_source.SearchRadius<Eigen::Vector3f>(
            key_points, radius, max_nn, indices_source, distance2_source);
    int result_target = kdtree_target.SearchRadius<Eigen::Vector3f>(
            key_points, radius, max_nn, indices_target, distance2_target);

    cupoch::utility::LogDebug("Trial example took : {:f}", ms_double.count());

    *result = *source + *target;

    cupoch::visualization::DrawGeometries({result, uniformSampled}, "Copoch",
                                          640, 480, 50, 50, true, true, false);
    return 0;
}
