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
#include "cupoch/kinematics/kinematic_chain.h"

int main(int argc, char *argv[]) {
    using namespace cupoch;
    utility::InitializeAllocator();

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 2) {utility::LogInfo("Need an argument of point cloud file name."); return 0;}
    auto kin = kinematics::KinematicChain(argv[1]);
    auto poses = kin.ForwardKinematics();
    auto meshes = kin.GetTransformedVisualGeometryMap(poses);
    std::vector<std::shared_ptr<const geometry::Geometry>> geoms;
    for (const auto& m: meshes) { geoms.emplace_back(m.second); };
    visualization::DrawGeometries(geoms);
    return 0;
}