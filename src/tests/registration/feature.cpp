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
#include "cupoch/registration/feature.h"

#include "cupoch/geometry/pointcloud.h"
#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

TEST(Feature, ComputeFPFHFeature) {
    std::vector<Vector3f> points;
    points.push_back(Vector3f(1.9765625, 2.58410227, 1.16015625));
    points.push_back(Vector3f(1.9609375, 2.58455765, 1.16015625));
    points.push_back(Vector3f(1.8984375, 2.58400571, 1.16015625));
    points.push_back(Vector3f(2.78255208, 2.58296331, 1.09896843));
    points.push_back(Vector3f(2.64478892, 2.59234305, 1.12572353));
    points.push_back(Vector3f(2.59420614, 2.59216719, 1.11943672));
    points.push_back(Vector3f(1.734375, 2.5838958, 1.16015625));
    points.push_back(Vector3f(1.69335938, 2.58432162, 1.16015625));
    points.push_back(Vector3f(1.65039062, 2.58413213, 1.16015625));
    points.push_back(Vector3f(1.53515625, 2.58283651, 1.15234375));
    points.push_back(Vector3f(2.49199047, 2.58972794, 1.11601081));
    points.push_back(Vector3f(2.77296469, 2.58743146, 1.08203125));
    points.push_back(Vector3f(2.09244757, 2.59142357, 1.12341505));
    points.push_back(Vector3f(2.04382068, 2.59081675, 1.1263028));
    points.push_back(Vector3f(2.19512649, 2.59046347, 1.12422752));
    points.push_back(Vector3f(2.19708807, 2.59833865, 1.09903618));
    points.push_back(Vector3f(2.09609375, 2.59854081, 1.09943067));
    points.push_back(Vector3f(1.44605704, 2.58916398, 1.12032112));
    points.push_back(Vector3f(1.89438566, 2.59281059, 1.12763938));
    points.push_back(Vector3f(1.84362796, 2.59052256, 1.12894744));

    std::vector<Vector3f> normals;
    normals.push_back(
            Vector3f(-1.72579880e-02, -9.57760371e-01, -2.87049184e-01));
    normals.push_back(
            Vector3f(-1.62135732e-02, -9.56575314e-01, -2.91033999e-01));
    normals.push_back(
            Vector3f(3.06446984e-01, -8.77611819e-01, -3.68629544e-01));
    normals.push_back(
            Vector3f(-1.76496988e-01, -9.69491985e-01, -1.70100274e-01));
    normals.push_back(
            Vector3f(6.23075953e-03, -9.45749699e-01, -3.24836397e-01));
    normals.push_back(
            Vector3f(3.28301903e-02, -9.39864091e-01, -3.39967159e-01));
    normals.push_back(
            Vector3f(-3.74394819e-02, -9.53596884e-01, -2.98749508e-01));
    normals.push_back(
            Vector3f(-1.03100234e-02, -9.60276075e-01, -2.78861191e-01));
    normals.push_back(
            Vector3f(9.44491594e-03, -9.61033596e-01, -2.76270195e-01));
    normals.push_back(
            Vector3f(2.70285700e-02, -9.60594043e-01, -2.76637922e-01));
    normals.push_back(
            Vector3f(3.55610463e-02, -9.45353416e-01, -3.24102347e-01));
    normals.push_back(
            Vector3f(-1.81109373e-01, -9.71147919e-01, -1.55148687e-01));
    normals.push_back(
            Vector3f(2.87692978e-04, -9.57385526e-01, -2.88812867e-01));
    normals.push_back(
            Vector3f(-2.38075871e-02, -9.58049272e-01, -2.85613009e-01));
    normals.push_back(
            Vector3f(-2.01180715e-02, -9.59536195e-01, -2.80865722e-01));
    normals.push_back(
            Vector3f(-2.41292450e-02, -9.60645238e-01, -2.76728217e-01));
    normals.push_back(
            Vector3f(-1.29694794e-03, -9.57518086e-01, -2.88370307e-01));
    normals.push_back(
            Vector3f(3.88871939e-02, -9.60903442e-01, -2.74139308e-01));
    normals.push_back(
            Vector3f(2.34180042e-01, -6.25141738e-01, -7.44551889e-01));
    normals.push_back(
            Vector3f(-2.50834290e-01, -7.19549596e-01, -6.47557363e-01));
    geometry::PointCloud pc;
    pc.SetPoints(points);
    pc.SetNormals(normals);

    Matrix<float, 33, 20> mat_features;
    mat_features << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49.24339957, 49.74101224,
            56.82796774, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.44077365, 0, 0, 0, 0,
            32.70366289, 18.09688723, 147.56842953, 145.57340304, 52.78273407,
            200, 200, 200, 192.88159149, 200, 200, 200, 200, 200, 200,
            192.55922635, 200, 200, 200, 200, 70.92132169, 66.18269826,
            3.1881709, 4.68558472, 90.38929819, 0, 0, 0, 7.11840851, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 96.37501542, 115.72041451, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            48.83713628, 48.92482492, 35.08156491, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            7.44077365, 0, 0, 0, 0, 54.90926126, 26.95924872, 0, 0, 4.04523367,
            0, 0, 0, 2.37280284, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.19353793,
            25, 49.24339957, 49.74101224, 56.82796774, 100.32930761,
            77.38421212, 53.40787966, 0, 0, 0, 0, 0, 100.35423858, 0,
            7.44077365, 0, 0, 0, 0, 32.70366289, 18.09688723, 101.22398725,
            100.36681348, 81.82796774, 99.67069239, 122.61578788, 146.59212034,
            159.54825816, 182.55929305, 193.76816635, 200, 200, 99.64576142,
            200, 185.11845269, 200, 200, 200, 200, 68.80646207, 64.25226068,
            0.6954769, 0.96734935, 18.17203226, 0, 0, 0, 2.37280284, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 31.19353793, 37.395694, 0, 0, 4.04523367, 0, 0,
            0, 35.70613617, 17.44070695, 6.23183365, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            6.19353793, 28.29590936, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.492694,
            3.71823537, 68.17203226, 0, 0, 0, 2.37280284, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 58.98793956, 53.32472051, 0, 0, 4.04523367, 0, 0, 0,
            2.37280284, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.19353793, 25,
            0.6954769, 0.96734935, 18.17203226, 0, 0, 0, 2.37280284, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 31.19353793, 37.395694, 147.56842953,
            145.57340304, 48.73750039, 99.67069239, 122.61578788, 146.59212034,
            157.17545532, 182.55929305, 193.76816635, 200, 200, 99.64576142,
            200, 192.55922635, 200, 200, 200, 200, 64.72778377, 37.8867889, 0,
            0, 4.04523367, 100.32930761, 77.38421212, 53.40787966, 35.70613617,
            17.44070695, 6.23183365, 0, 0, 100.35423858, 0, 0, 0, 0, 0, 0,
            6.19353793, 28.29590936, 49.24339957, 49.74101224, 56.82796774, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 7.44077365, 0, 0, 0, 0, 32.70366289,
            18.09688723, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    std::vector<registration::Feature<33>::FeatureType> ref_features;
    for (int i = 0; i < mat_features.cols(); i++) {
        ref_features.push_back(mat_features.col(i));
    }
    auto features = registration::ComputeFPFHFeature(
            pc, knn::KDTreeSearchParamRadius(0.05 * 3, 5));
    ExpectEQ(ref_features, features->GetData());
}