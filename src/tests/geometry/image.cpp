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
#include "cupoch/geometry/image.h"

#include "cupoch/camera/pinhole_camera_intrinsic.h"
#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

using ConversionType = geometry::Image::ColorToIntensityConversionType;
using FilterType = geometry::Image::FilterType;

TEST(Image, DefaultConstructor) {
    geometry::Image image;

    // inherited from Geometry2D
    EXPECT_EQ(geometry::Geometry::GeometryType::Image, image.GetGeometryType());
    EXPECT_EQ(2, image.Dimension());

    // public member variables
    EXPECT_EQ(0, image.width_);
    EXPECT_EQ(0, image.height_);
    EXPECT_EQ(0, image.num_of_channels_);
    EXPECT_EQ(0, image.bytes_per_channel_);
    EXPECT_EQ(0u, image.data_.size());

    // public members
    EXPECT_TRUE(image.IsEmpty());
    EXPECT_FALSE(image.HasData());

    ExpectEQ(Zero2f, image.GetMinBound());
    ExpectEQ(Zero2f, image.GetMaxBound());

    EXPECT_FALSE(image.TestImageBoundary(0, 0));
    EXPECT_EQ(0, image.BytesPerLine());
}

TEST(Image, CreateImage) {
    int width = 1920;
    int height = 1080;
    int num_of_channels = 3;
    int bytes_per_channel = 1;

    geometry::Image image;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    // public member variables
    EXPECT_EQ(width, image.width_);
    EXPECT_EQ(height, image.height_);
    EXPECT_EQ(num_of_channels, image.num_of_channels_);
    EXPECT_EQ(bytes_per_channel, image.bytes_per_channel_);
    EXPECT_EQ(size_t(width * height * num_of_channels * bytes_per_channel),
              image.data_.size());

    // public members
    EXPECT_FALSE(image.IsEmpty());
    EXPECT_TRUE(image.HasData());

    ExpectEQ(Zero2f, image.GetMinBound());
    ExpectEQ(Vector2f(width, height), image.GetMaxBound());

    EXPECT_TRUE(image.TestImageBoundary(0, 0));
    EXPECT_EQ(width * num_of_channels * bytes_per_channel,
              image.BytesPerLine());
}

TEST(Image, Clear) {
    int width = 1920;
    int height = 1080;
    int num_of_channels = 3;
    int bytes_per_channel = 1;

    geometry::Image image;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    image.Clear();

    // public member variables
    EXPECT_EQ(0, image.width_);
    EXPECT_EQ(0, image.height_);
    EXPECT_EQ(0, image.num_of_channels_);
    EXPECT_EQ(0, image.bytes_per_channel_);
    EXPECT_EQ(0u, image.data_.size());

    // public members
    EXPECT_TRUE(image.IsEmpty());
    EXPECT_FALSE(image.HasData());

    ExpectEQ(Zero2f, image.GetMinBound());
    ExpectEQ(Zero2f, image.GetMaxBound());

    EXPECT_FALSE(image.TestImageBoundary(0, 0));
    EXPECT_EQ(0, image.BytesPerLine());
}

TEST(Image, FloatValueAt) {
    geometry::Image image;

    int width = 10;
    int height = 10;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);
    thrust::host_vector<uint8_t> h_data = image.GetData();

    float* const im = Cast<float>(&h_data[0]);

    im[0 * width + 0] = 4.0f;
    im[0 * width + 1] = 4.0f;
    im[1 * width + 0] = 4.0f;
    im[1 * width + 1] = 4.0f;

    EXPECT_NEAR(4.0f,
                geometry::FloatValueAt(h_data.data(), 0.0, 0.0, width, height,
                                       num_of_channels, bytes_per_channel)
                        .second,
                THRESHOLD_1E_4);
    EXPECT_NEAR(4.0f,
                geometry::FloatValueAt(h_data.data(), 0.0, 1.0, width, height,
                                       num_of_channels, bytes_per_channel)
                        .second,
                THRESHOLD_1E_4);
    EXPECT_NEAR(4.0f,
                geometry::FloatValueAt(h_data.data(), 1.0, 0.0, width, height,
                                       num_of_channels, bytes_per_channel)
                        .second,
                THRESHOLD_1E_4);
    EXPECT_NEAR(4.0f,
                geometry::FloatValueAt(h_data.data(), 1.0, 1.0, width, height,
                                       num_of_channels, bytes_per_channel)
                        .second,
                THRESHOLD_1E_4);
    EXPECT_NEAR(4.0f,
                geometry::FloatValueAt(h_data.data(), 0.5, 0.5, width, height,
                                       num_of_channels, bytes_per_channel)
                        .second,
                THRESHOLD_1E_4);
    EXPECT_NEAR(2.0f,
                geometry::FloatValueAt(h_data.data(), 0.0, 1.5, width, height,
                                       num_of_channels, bytes_per_channel)
                        .second,
                THRESHOLD_1E_4);
    EXPECT_NEAR(2.0f,
                geometry::FloatValueAt(h_data.data(), 1.5, 0.0, width, height,
                                       num_of_channels, bytes_per_channel)
                        .second,
                THRESHOLD_1E_4);
    EXPECT_NEAR(1.0f,
                geometry::FloatValueAt(h_data.data(), 1.5, 1.5, width, height,
                                       num_of_channels, bytes_per_channel)
                        .second,
                THRESHOLD_1E_4);
}

TEST(Image, CreateDepthToCameraDistanceMultiplierFloatImage) {
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    auto image =
            geometry::Image::CreateDepthToCameraDistanceMultiplierFloatImage(
                    intrinsic);

    // test image dimensions
    int width = 640;
    int height = 480;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    EXPECT_FALSE(image->IsEmpty());
    EXPECT_EQ(width, image->width_);
    EXPECT_EQ(height, image->height_);
    EXPECT_EQ(num_of_channels, image->num_of_channels_);
    EXPECT_EQ(bytes_per_channel, image->bytes_per_channel_);
}

void TEST_CreateFloatImage(
        const int& num_of_channels,
        const int& bytes_per_channel,
        const std::vector<uint8_t>& ref,
        const geometry::Image::ColorToIntensityConversionType& type) {
    geometry::Image image;

    // test image dimensions
    int width = 5;
    int height = 5;
    int float_num_of_channels = 1;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    std::vector<uint8_t> data(image.data_.size());
    Rand(data, 0, 255, 0);
    image.SetData(data);

    auto float_image = image.CreateFloatImage();

    EXPECT_FALSE(float_image->IsEmpty());
    EXPECT_EQ(width, float_image->width_);
    EXPECT_EQ(height, float_image->height_);
    EXPECT_EQ(float_num_of_channels, float_image->num_of_channels_);
    EXPECT_EQ(int(sizeof(float)), float_image->bytes_per_channel_);
    ExpectEQ(ref, float_image->GetData());
}

TEST(Image, CreateFloatImage_1_1) {
    // reference data used to validate the creation of the float image
    uint8_t raw_ref[] = {
            216, 214, 86,  63,  202, 200, 200, 62,  201, 199, 71,  63,  205,
            203, 75,  63,  234, 232, 104, 63,  202, 200, 72,  62,  171, 170,
            170, 62,  197, 195, 67,  63,  141, 140, 140, 62,  142, 141, 13,
            63,  244, 242, 242, 62,  161, 160, 32,  63,  187, 186, 186, 62,
            131, 130, 2,   63,  244, 242, 114, 63,  235, 233, 105, 63,  163,
            162, 34,  63,  183, 182, 54,  63,  145, 144, 16,  62,  155, 154,
            26,  63,  129, 128, 128, 60,  246, 244, 116, 62,  137, 136, 8,
            62,  207, 205, 77,  63,  157, 156, 28,  62};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_CreateFloatImage(1, 1, ref, ConversionType::Weighted);
}

TEST(Image, CreateFloatImage_1_2) {
    // reference data used to validate the creation of the float image
    uint8_t raw_ref[] = {
            0, 172, 201, 70, 0, 199, 75,  71, 0, 160, 75,  70, 0, 85,  67,  71,
            0, 70,  13,  71, 0, 121, 32,  71, 0, 93,  2,   71, 0, 242, 105, 71,
            0, 162, 54,  71, 0, 36,  26,  71, 0, 16,  116, 70, 0, 34,  77,  71,
            0, 78,  204, 70, 0, 8,   217, 69, 0, 248, 95,  70, 0, 130, 85,  71,
            0, 56,  151, 70, 0, 162, 5,   71, 0, 125, 120, 71, 0, 74,  68,  71,
            0, 134, 68,  71, 0, 102, 99,  71, 0, 144, 178, 70, 0, 205, 106, 71,
            0, 17,  114, 71};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_CreateFloatImage(1, 2, ref, ConversionType::Weighted);
}

TEST(Image, CreateFloatImage_1_4) {
    // reference data used to validate the creation of the float image
    uint8_t raw_ref[] = {
            214, 100, 199, 203, 232, 50,  85,  195, 70,  141, 121, 160, 93,
            130, 242, 233, 162, 182, 36,  154, 4,   61,  34,  205, 39,  102,
            33,  27,  254, 55,  130, 213, 156, 75,  162, 133, 125, 248, 74,
            196, 134, 196, 102, 227, 72,  89,  205, 234, 17,  242, 134, 21,
            49,  169, 227, 88,  16,  5,   116, 16,  60,  247, 230, 216, 67,
            137, 95,  193, 130, 170, 135, 10,  111, 237, 237, 183, 72,  188,
            163, 90,  175, 42,  112, 224, 211, 84,  58,  227, 89,  175, 243,
            150, 167, 218, 112, 235, 101, 207, 174, 232};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_CreateFloatImage(1, 4, ref, ConversionType::Weighted);
}

TEST(Image, CreateFloatImage_3_1_Weighted) {
    // reference data used to validate the creation of the float image
    uint8_t raw_ref[] = {
            45,  241, 17,  63,  30,  96,  75,  63,  154, 112, 20,  63,  0,
            241, 3,   63,  180, 56,  4,   63,  139, 60,  58,  63,  116, 8,
            204, 62,  216, 59,  119, 62,  65,  47,  151, 62,  252, 20,  36,
            63,  194, 101, 54,  63,  138, 51,  5,   63,  55,  35,  64,  63,
            95,  59,  32,  63,  29,  161, 44,  63,  137, 77,  46,  63,  199,
            12,  35,  63,  122, 21,  90,  62,  101, 168, 243, 62,  209, 97,
            143, 62,  10,   228, 61,  63,  224, 255, 239, 62,  59,  33,  29,
            63,  197, 186, 3,   63,  145, 27,  72,  63};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_CreateFloatImage(3, 1, ref, ConversionType::Weighted);
}

TEST(Image, CreateFloatImage_3_1_Equal) {
    // reference data used to validate the creation of the float image
    uint8_t raw_ref[] = {
            45,  241, 17,  63,  30,  96,  75,  63,  154, 112, 20,  63,  0,
            241, 3,   63,  180, 56,  4,   63,  139, 60,  58,  63,  116, 8,
            204, 62,  216, 59,  119, 62,  65,  47,  151, 62,  252, 20,  36,
            63,  194, 101, 54,  63,  138, 51,  5,   63,  55,  35,  64,  63,
            95,  59,  32,  63,  29,  161, 44,  63,  137, 77,  46,  63,  199,
            12,  35,  63,  122, 21,  90,  62,  101, 168, 243, 62,  209, 97,
            143, 62,  10,   228, 61,  63,  224, 255, 239, 62,  59,  33,  29,
            63,  197, 186, 3,   63,  145, 27,  72,  63};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_CreateFloatImage(3, 1, ref, ConversionType::Equal);
}

TEST(Image, CreateFloatImage_3_2_Weighted) {
    // reference data used to validate the creation of the float image
    uint8_t raw_ref[] = {
            16,  146, 27,  71,  44,  160, 31,  71,  234, 31,  69,  71,  39,
            148, 210, 70,  195, 103, 83,  70,  79,  233, 246, 70,  96,  236,
            83,  71,  227, 42,  19,  71,  145, 153, 208, 70,  82,  101, 251,
            69,  235, 227, 88,  71,  46,  27,  31,  71,  208, 107, 72,  71,
            169, 123, 155, 70,  236, 187, 50,  71,  151, 82,  72,  71,  49,
            235, 76,  71,  31,  111, 86,  71,  27,  105, 148, 70,  71,  196,
            219, 70,  12,  108, 22,  71,  197, 41,  183, 70,  225, 5,   23,
            71,  210, 181, 85,  71,  101, 14,  28,  71};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_CreateFloatImage(3, 2, ref, ConversionType::Weighted);
}

TEST(Image, CreateFloatImage_3_2_Equal) {
    // reference data used to validate the creation of the float image
    uint8_t raw_ref[] = {
            16,  146, 27,  71,  44,  160, 31,  71,  234, 31,  69,  71,  39,
            148, 210, 70,  195, 103, 83,  70,  79,  233, 246, 70,  96,  236,
            83,  71,  227, 42,  19,  71,  145, 153, 208, 70,  82,  101, 251,
            69,  235, 227, 88,  71,  46,  27,  31,  71,  208, 107, 72,  71,
            169, 123, 155, 70,  236, 187, 50,  71,  151, 82,  72,  71,  49,
            235, 76,  71,  31,  111, 86,  71,  27,  105, 148, 70,  71,  196,
            219, 70,  12,  108, 22,  71,  197, 41,  183, 70,  225, 5,   23,
            71,  210, 181, 85,  71,  101, 14,  28,  71};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_CreateFloatImage(3, 2, ref, ConversionType::Equal);
}

TEST(Image, CreateFloatImage_3_4_Weighted) {
    // reference data used to validate the creation of the float image
    uint8_t raw_ref[] = {
            154, 122, 238, 202, 65,  5,   17,  233, 117, 224, 24,  213, 167,
            79,  59,  233, 15,  163, 133, 88,  22,  30,  10,  216, 24,  168,
            218, 222, 111, 170, 219, 233, 198, 232, 16,  109, 227, 84,  156,
            229, 56,  95,  77,  97,  226, 226, 200, 188, 36,  128, 64,  193,
            178, 161, 146, 208, 240, 239, 83,  208, 189, 119, 176, 114, 209,
            111, 82,  249, 14,  45,  72,  210, 222, 97,  25,  247, 179, 223,
            15,  114, 245, 201, 149, 76,  224, 3,   24,  64,  17,  103, 98,
            222, 145, 236, 94,  233, 36,  85,  141, 233};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_CreateFloatImage(3, 4, ref, ConversionType::Weighted);
}

TEST(Image, CreateFloatImage_3_4_Equal) {
    // reference data used to validate the creation of the float image
    uint8_t raw_ref[] = {
            154, 122, 238, 202, 65,  5,   17,  233, 117, 224, 24,  213, 167,
            79,  59,  233, 15,  163, 133, 88,  22,  30,  10,  216, 24,  168,
            218, 222, 111, 170, 219, 233, 198, 232, 16,  109, 227, 84,  156,
            229, 56,  95,  77,  97,  226, 226, 200, 188, 36,  128, 64,  193,
            178, 161, 146, 208, 240, 239, 83,  208, 189, 119, 176, 114, 209,
            111, 82,  249, 14,  45,  72,  210, 222, 97,  25,  247, 179, 223,
            15,  114, 245, 201, 149, 76,  224, 3,   24,  64,  17,  103, 98,
            222, 145, 236, 94,  233, 36,  85,  141, 233};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_CreateFloatImage(3, 4, ref, ConversionType::Equal);
}

TEST(Image, ConvertDepthToFloatImage) {
    // reference data used to validate the creation of the float image
    uint8_t raw_ref[] = {
            210, 254, 91,  58,  105, 154, 205, 57,  61,  147, 76,  58,  237,
            175, 80,  58,  234, 127, 110, 58,  105, 154, 77,  57,  63,  195,
            174, 57,  141, 118, 72,  58,  22,  236, 143, 57,  66,  243, 16,
            58,  163, 199, 248, 57,  135, 123, 36,  58,  0, 54,  191, 57,
            94,  164, 5,   58,  163, 199, 120, 58,  22,  135, 111, 58,  223,
            137, 38,  58,  79,  25,  59,  58,  198, 8,   20,  57,  126, 80,
            30,  58,  6,   150, 131, 55,  251, 213, 122, 57,  102, 207, 11,
            57,  69,  190, 82,  58,  215, 94,  32,  57};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);
    geometry::Image image;

    // test image dimensions
    int width = 5;
    int height = 5;
    int num_of_channels = 1;
    int bytes_per_channel = 1;
    int float_num_of_channels = 1;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    std::vector<uint8_t> data(image.data_.size());
    Rand(data, 0, 255, 0);
    image.SetData(data);

    auto float_image = image.ConvertDepthToFloatImage();

    EXPECT_FALSE(float_image->IsEmpty());
    EXPECT_EQ(width, float_image->width_);
    EXPECT_EQ(height, float_image->height_);
    EXPECT_EQ(float_num_of_channels, float_image->num_of_channels_);
    EXPECT_EQ(int(sizeof(float)), float_image->bytes_per_channel_);
    ExpectEQ(ref, float_image->GetData());
}

TEST(Image, TransposeUint8) {
    // reference data used to validate the creation of the float image
    // clang-format off
    uint8_t raw_input[] = {
        0,  1,  2,  3,  4,  5,
        6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17
    };
    std::vector<uint8_t> input;
    for (int i = 0; i < 18; ++i) input.push_back(raw_input[i]);
    uint8_t raw_transposed_ref[] = {
        0,  6,  12,
        1,  7,  13,
        2,  8,  14,
        3,  9,  15,
        4,  10, 16,
        5,  11, 17
    };
    std::vector<uint8_t> transposed_ref;
    for (int i = 0; i < 18; ++i) transposed_ref.push_back(raw_transposed_ref[i]);
    // clang-format on

    geometry::Image image;

    int width = 6;
    int height = 3;
    int num_of_channels = 1;
    int bytes_per_channel = 1;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);
    image.SetData(input);

    auto transposed_image = image.Transpose();
    EXPECT_FALSE(transposed_image->IsEmpty());
    EXPECT_EQ(height, transposed_image->width_);
    EXPECT_EQ(width, transposed_image->height_);
    EXPECT_EQ(num_of_channels, transposed_image->num_of_channels_);
    EXPECT_EQ(int(sizeof(uint8_t)), transposed_image->bytes_per_channel_);
    ExpectEQ(transposed_ref, transposed_image->GetData());
}

TEST(Image, TransposeFloat) {
    // reference data used to validate the creation of the float image
    // clang-format off
    float raw_input[] = {
        0,  1,  2,  3,  4,  5,
        6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17
    };
    std::vector<float> input;
    for (int i = 0; i < 18; ++i) input.push_back(raw_input[i]);
    float raw_transposed_ref[] = {
        0,  6,  12,
        1,  7,  13,
        2,  8,  14,
        3,  9,  15,
        4,  10, 16,
        5,  11, 17
    };
    std::vector<float> transposed_ref;
    for (int i = 0; i < 18; ++i) transposed_ref.push_back(raw_transposed_ref[i]);
    // clang-format on

    geometry::Image image;

    int width = 6;
    int height = 3;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);
    const uint8_t* input_uint8_ptr =
            reinterpret_cast<const uint8_t*>(input.data());
    std::vector<uint8_t> input_uint8(
            input_uint8_ptr, input_uint8_ptr + image.data_.size());
    image.SetData(input_uint8);

    auto transposed_image = image.Transpose();
    EXPECT_FALSE(transposed_image->IsEmpty());
    EXPECT_EQ(height, transposed_image->width_);
    EXPECT_EQ(width, transposed_image->height_);
    EXPECT_EQ(num_of_channels, transposed_image->num_of_channels_);
    EXPECT_EQ(int(sizeof(float)), transposed_image->bytes_per_channel_);

    std::vector<uint8_t> transposed_host_data =
            transposed_image->GetData();
    const float* transpose_image_floats =
            reinterpret_cast<const float*>(transposed_host_data.data());
    std::vector<float> transpose_image_data(
            transpose_image_floats,
            transpose_image_floats + transposed_ref.size());
    ExpectEQ(transposed_ref, transpose_image_data);
}

TEST(Image, FlipVerticalImage) {
    // reference data used to validate the creation of the float image
    // clang-format off
    uint8_t raw_input[] = {
      0, 1, 2, 3, 4, 5,
      6, 7, 8, 9, 10, 11,
      12, 13, 14, 15, 16, 17
    };
    std::vector<uint8_t> input;
    for (int i = 0; i < 18; ++i) input.push_back(raw_input[i]);
    uint8_t raw_flipped[] = {
      12, 13, 14, 15, 16, 17,
      6, 7, 8, 9, 10, 11,
      0, 1, 2, 3, 4, 5,
    };
    std::vector<uint8_t> flipped;
    for (int i = 0; i < 18; ++i) flipped.push_back(raw_flipped[i]);
    // clang-format on

    geometry::Image image;
    // test image dimensions
    int width = 6;
    int height = 3;
    int num_of_channels = 1;
    int bytes_per_channel = 1;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);
    image.SetData(input);

    auto flip_image = image.FlipVertical();
    EXPECT_FALSE(flip_image->IsEmpty());
    EXPECT_EQ(width, flip_image->width_);
    EXPECT_EQ(height, flip_image->height_);
    EXPECT_EQ(num_of_channels, flip_image->num_of_channels_);
    EXPECT_EQ(int(sizeof(uint8_t)), flip_image->bytes_per_channel_);
    ExpectEQ(flipped, flip_image->GetData());
}

TEST(Image, FlipHorizontalImage) {
    // reference data used to validate the creation of the float image
    // clang-format off
    uint8_t raw_input[] = {
      0, 1, 2, 3, 4, 5,
      6, 7, 8, 9, 10, 11,
      12, 13, 14, 15, 16, 17
    };
    std::vector<uint8_t> input;
    for (int i = 0; i < 18; ++i) input.push_back(raw_input[i]);
    uint8_t raw_flipped[] = {
      5, 4, 3, 2, 1, 0,
      11, 10, 9, 8, 7, 6,
      17, 16, 15, 14, 13, 12
    };
    std::vector<uint8_t> flipped;
    for (int i = 0; i < 18; ++i) flipped.push_back(raw_flipped[i]);
    // clang-format on

    geometry::Image image;
    // test image dimensions
    int width = 6;
    int height = 3;
    int num_of_channels = 1;
    int bytes_per_channel = 1;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);
    image.SetData(input);

    auto flip_image = image.FlipHorizontal();
    EXPECT_FALSE(flip_image->IsEmpty());
    EXPECT_EQ(width, flip_image->width_);
    EXPECT_EQ(height, flip_image->height_);
    EXPECT_EQ(num_of_channels, flip_image->num_of_channels_);
    EXPECT_EQ(int(sizeof(uint8_t)), flip_image->bytes_per_channel_);
    ExpectEQ(flipped, flip_image->GetData());
}

void TEST_Filter(const std::vector<uint8_t>& ref,
                 const geometry::Image::FilterType& filter) {
    geometry::Image image;

    // test image dimensions
    int width = 5;
    int height = 5;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    std::vector<uint8_t> data(image.data_.size());
    Rand(data, 0, 255, 0);
    image.SetData(data);

    auto float_image = image.CreateFloatImage();

    auto output = float_image->Filter(filter);

    EXPECT_FALSE(output->IsEmpty());
    EXPECT_EQ(width, output->width_);
    EXPECT_EQ(height, output->height_);
    EXPECT_EQ(num_of_channels, output->num_of_channels_);
    EXPECT_EQ(bytes_per_channel, output->bytes_per_channel_);
    ExpectEQ(ref, output->GetData());
}

TEST(Image, Filter_Gaussian3) {
    // reference data used to validate the filtering of an image
    uint8_t raw_ref[] = {
            41,  194, 49,  204, 116, 56,  130, 211, 198, 225, 181, 232, 198,
            225, 53,  233, 198, 225, 181, 232, 177, 94,  205, 232, 47,  90,
            77,  233, 240, 252, 4,   233, 93,  130, 114, 232, 93,  130, 242,
            231, 177, 94,  77,  233, 47,  90,  205, 233, 72,  89,  77,  233,
            6,   134, 220, 88,  128, 234, 129, 89,  60,  96,  205, 232, 167,
            91,  77,  233, 2,   196, 171, 233, 229, 149, 243, 233, 12,  159,
            128, 233, 36,  49,  20,  226, 223, 39,  141, 226, 137, 164, 52,
            234, 108, 176, 182, 234, 146, 238, 64,  234};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_Filter(ref, FilterType::Gaussian3);
}

TEST(Image, Filter_Gaussian5) {
    // reference data used to validate the filtering of an image
    uint8_t raw_ref[] = {
            61,  94,  205, 231, 230, 96,  109, 232, 15,  16,  218, 232, 2,
            118, 3,   233, 160, 185, 166, 232, 61,  94,  205, 232, 46,  125,
            35,  233, 60,  145, 12,  233, 110, 3,   165, 232, 122, 145, 23,
            232, 223, 6,   26,  233, 24,  249, 119, 233, 159, 37,  94,  233,
            235, 229, 13,  233, 99,  24,  143, 232, 40,  96,  205, 232, 206,
            73,  101, 233, 15,  186, 202, 233, 62,  231, 242, 233, 76,  236,
            159, 233, 35,  111, 205, 231, 102, 26,  76,  233, 254, 241, 44,
            234, 33,  174, 126, 234, 84,  234, 47,  234};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_Filter(ref, FilterType::Gaussian5);
}

TEST(Image, Filter_Gaussian7) {
    // reference data used to validate the filtering of an image
    uint8_t raw_ref[] = {
            71,  19,  68,  232, 29,  11,  169, 232, 178, 140, 214, 232, 35,
            21,  214, 232, 245, 42,  147, 232, 65,  168, 175, 232, 125, 101,
            5,   233, 242, 119, 15,  233, 60,  92,  246, 232, 131, 231, 154,
            232, 225, 75,  240, 232, 84,  18,  69,  233, 128, 68,  108, 233,
            67,  141, 98,  233, 63,  199, 27,  233, 109, 191, 244, 232, 122,
            49,  127, 233, 20,  166, 194, 233, 176, 46,  222, 233, 33,  207,
            168, 233, 186, 237, 232, 232, 98,  40,  161, 233, 128, 206, 18,
            234, 109, 135, 55,  234, 187, 97,  17,  234};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_Filter(ref, FilterType::Gaussian7);
}

TEST(Image, Filter_Sobel3Dx) {
    // reference data used to validate the filtering of an image
    uint8_t raw_ref[] = {
            172, 2,   109, 77,  136, 55,  130, 213, 198, 225, 181, 234, 254,
            55,  130, 85,  198, 225, 181, 106, 122, 87,  205, 234, 134, 196,
            102, 99,  177, 184, 144, 106, 254, 55,  2,   86,  93,  130, 242,
            105, 122, 87,  77,  235, 138, 196, 230, 99,  72,  89,  77,  107,
            214, 220, 163, 90,  34,  71,  135, 90,  231, 88,  205, 234, 63,
            133, 106, 99,  73,  45,  10,  235, 101, 207, 174, 232, 44,  100,
            107, 107, 28,  239, 8,   228, 119, 32,  52,  97,  114, 163, 52,
            236, 140, 27,  131, 233, 33,  139, 48,  108};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_Filter(ref, FilterType::Sobel3Dx);
}

TEST(Image, Filter_Sobel3Dy) {
    // reference data used to validate the filtering of an image
    uint8_t raw_ref[] = {
            151, 248, 205, 205, 67,  56,  130, 213, 93,  130, 242, 105, 93,
            130, 114, 106, 93,  130, 242, 105, 177, 94,  205, 234, 47,  90,
            77,  235, 177, 184, 144, 234, 93,  130, 114, 106, 93,  130, 242,
            105, 108, 57,  173, 217, 91,  238, 228, 216, 254, 55,  2,   86,
            214, 220, 163, 90,  108, 154, 117, 91,  38,  93,  205, 106, 183,
            88,  77,  107, 189, 46,  10,  235, 229, 149, 243, 235, 12,  159,
            128, 235, 189, 150, 69,  227, 36,  53,  188, 227, 97,  219, 112,
            235, 229, 149, 243, 235, 12,  159, 128, 235};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_Filter(ref, FilterType::Sobel3Dy);
}

TEST(Image, FilterHorizontal) {
    // reference data used to validate the filtering of an image
    uint8_t raw_ref[] = {
            187, 139, 149, 203, 171, 101, 199, 202, 93,  130, 242, 232, 93,
            130, 114, 233, 93,  130, 242, 232, 134, 91,  243, 204, 79,  56,
            130, 212, 254, 55,  2,   213, 254, 55,  130, 212, 94,  58,  24,
            196, 177, 94,  205, 233, 47,  90,  77,  234, 72,  89,  205, 233,
            49,  169, 99,  88,  49,  169, 227, 87,  109, 57,  173, 216, 60,
            247, 230, 215, 97,  137, 95,  192, 72,  188, 163, 89,  108, 154,
            117, 90,  211, 150, 69,  226, 40,  53,  188, 226, 97,  219, 112,
            234, 229, 149, 243, 234, 12,  159, 128, 234};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    geometry::Image image;

    // test image dimensions
    int width = 5;
    int height = 5;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    std::vector<uint8_t> data(image.data_.size());
    Rand(data, 0, 255, 0);
    image.SetData(data);

    auto float_image = image.CreateFloatImage();

    std::vector<float> Gaussian3(3);
    Gaussian3[0] = 0.25;
    Gaussian3[1] = 0.5;
    Gaussian3[2] = 0.25;

    auto output = float_image->FilterHorizontal(Gaussian3);

    EXPECT_FALSE(output->IsEmpty());
    EXPECT_EQ(width, output->width_);
    EXPECT_EQ(height, output->height_);
    EXPECT_EQ(num_of_channels, output->num_of_channels_);
    EXPECT_EQ(bytes_per_channel, output->bytes_per_channel_);
    ExpectEQ(ref, output->GetData());
}

TEST(Image, Downsample) {
    // reference data used to validate the filtering of an image
    uint8_t raw_ref[] = {172, 41, 59,  204, 93, 130, 242, 232,
                         22,  91, 205, 233, 49, 169, 227, 87};
    std::vector<uint8_t> ref;
    for (int i = 0; i < 16; ++i) ref.push_back(raw_ref[i]);

    geometry::Image image;

    // test image dimensions
    int width = 5;
    int height = 5;
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    std::vector<uint8_t> data(image.data_.size());
    Rand(data, 0, 255, 0);
    image.SetData(data);

    auto float_image = image.CreateFloatImage();

    auto output = float_image->Downsample();

    EXPECT_FALSE(output->IsEmpty());
    EXPECT_EQ((int)(width / 2), output->width_);
    EXPECT_EQ((int)(height / 2), output->height_);
    EXPECT_EQ(num_of_channels, output->num_of_channels_);
    EXPECT_EQ(bytes_per_channel, output->bytes_per_channel_);
    ExpectEQ(ref, output->GetData());
}