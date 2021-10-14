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
#include "cupoch/cupoch.h"

int main(int argc, char * argv[])
{
  using namespace cupoch;
  utility::InitializeAllocator();

  utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
  if (argc < 3) {utility::LogInfo("Need two arguments of point cloud file name."); return 0;}

  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;
  auto t1 = high_resolution_clock::now();

  auto source = std::make_shared<geometry::PointCloud>();
  auto target = std::make_shared<geometry::PointCloud>();
  auto result = std::make_shared<geometry::PointCloud>();
  if (io::ReadPointCloud(argv[1], *source)) {
    utility::LogInfo("Successfully read {}", argv[1]);
  } else {
    utility::LogWarning("Failed to read {}", argv[1]);
  }
  if (io::ReadPointCloud(argv[2], *target)) {
    utility::LogInfo("Successfully read {}", argv[2]);
  } else {
    utility::LogWarning("Failed to read {}", argv[2]);
  }
  
  Eigen::Matrix4f eye = Eigen::Matrix4f::Identity();
  cupoch::registration::ICPConvergenceCriteria criteria;
  criteria.max_iteration_ = 100;
  auto estimation =
    cupoch::registration::TransformationEstimationPointToPoint();

  auto res = registration::RegistrationICP(
    *source, *target, 0.2, eye, estimation, criteria);

  source->Transform(res.transformation_);

  auto t2 = high_resolution_clock::now();

  duration<double, std::milli> ms_double = t2 - t1;

  cupoch::utility::LogDebug("Registration of given clouds took {:f} ms", ms_double.count());

  visualization::DrawGeometries({source, target, result});
  return 0;
}
