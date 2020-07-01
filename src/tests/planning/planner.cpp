#include "cupoch/planning/planner.h"
#include "cupoch/geometry/boundingvolume.h"
#include "tests/test_utility/unit_test.h"

using namespace cupoch;
using namespace unit_test;

TEST(SimplePlanner, FindPath) {
    auto graph = geometry::Graph<3>::CreateFromAxisAlignedBoundingBox(geometry::AxisAlignedBoundingBox(Eigen::Vector3f(-1.0, -1.0, -1.0),
                                                                                                       Eigen::Vector3f(1.0, 1.0, 1.0)),
                                                                      Eigen::Vector3i(10, 10, 10));
    planning::Pos3DPlanner planner(*graph);
    planner.FindPath(Eigen::Vector3f(-1.0, -0.5, -0.2), Eigen::Vector3f(0.8, 0.9, 0.7));
}
