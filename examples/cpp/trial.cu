#include "cupoch/cupoch.h"

int main(int argc, char* argv[]) {
    using namespace cupoch;
    utility::InitializeAllocator();
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;
    auto t1 = high_resolution_clock::now();

    if (argc < 2) {
        utility::LogInfo("Need an argument of point cloud file name.");
        return 0;
    }
    auto pcd = io::CreatePointCloudFromFile(argv[1]);

    // remove the ground
    auto segmented = pcd->SegmentPlane(0.3, 3, 50);
    pcd = pcd->SelectByIndex(std::get<1>(segmented), true);

    // estimate normals
    pcd->EstimateNormals();

    utility::LogDebug("Going to Estimate FPFH  with {:d} points",
                      pcd->GetPoints().size());

    // compute fast point feature histograms
    auto fpfh = cupoch::registration::ComputeFPFHFeature(*pcd);
    auto shot = cupoch::registration::ComputeSHOTFeature(*pcd, 0.8);

    utility::LogDebug("Estimated FPFH are");
    std::cout << fpfh->GetData()[0] << std::endl;

    utility::LogDebug("Estimated SHOT are");
    std::cout << shot->GetData()[0] << std::endl;

    geometry::KDTreeFlann kdtree(*pcd);

    int max_nn = 8;
    float radius = 1.0;
    cupoch::utility::device_vector<int> indices;
    cupoch::utility::device_vector<float> distance2;

    cupoch::utility::device_vector<Eigen::Vector3f> q;
    q.push_back(Eigen::Vector3f(0, 0, 0));
    q.push_back(Eigen::Vector3f(0, 0, 0));

    int result = kdtree.SearchRadius<Eigen::Vector3f>(q, radius, max_nn,
                                                      indices, distance2);

    utility::LogDebug("Estimated SHOT are");
    std::cout << shot->GetData()[0] << std::endl;

    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    utility::LogDebug("trial example took : {:f}", ms_double.count());

    visualization::DrawGeometries({pcd}, "Copoch", 640, 480, 50, 50, true, true,
                                  false);
    return 0;
}
