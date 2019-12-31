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

void TEST_CreateFloatImage(
        const int& num_of_channels,
        const int& bytes_per_channel,
        const thrust::host_vector<uint8_t>& ref,
        const geometry::Image::ColorToIntensityConversionType& type) {
    geometry::Image image;

    // test image dimensions
    int width = 5;
    int height = 5;
    int float_num_of_channels = 1;

    image.Prepare(width, height, num_of_channels, bytes_per_channel);

    thrust::host_vector<uint8_t> data(image.data_.size());
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
            215, 214, 86,  63,  201, 200, 200, 62,  200, 199, 71,  63,  204,
            203, 75,  63,  233, 232, 104, 63,  201, 200, 72,  62,  171, 170,
            170, 62,  196, 195, 67,  63,  141, 140, 140, 62,  142, 141, 13,
            63,  243, 242, 242, 62,  161, 160, 32,  63,  187, 186, 186, 62,
            131, 130, 2,   63,  243, 242, 114, 63,  234, 233, 105, 63,  163,
            162, 34,  63,  183, 182, 54,  63,  145, 144, 16,  62,  155, 154,
            26,  63,  129, 128, 128, 60,  245, 244, 116, 62,  137, 136, 8,
            62,  206, 205, 77,  63,  157, 156, 28,  62};
    thrust::host_vector<uint8_t> ref;
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
    thrust::host_vector<uint8_t> ref;
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
    thrust::host_vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_CreateFloatImage(1, 4, ref, ConversionType::Weighted);
}

TEST(Image, CreateFloatImage_3_1_Weighted) {
    // reference data used to validate the creation of the float image
    uint8_t raw_ref[] = {
            44,  241, 17,  63,  29,  96,  75,  63,  154, 112, 20,  63,  255,
            240, 3,   63,  180, 56,  4,   63,  139, 60,  58,  63,  115, 8,
            204, 62,  215, 59,  119, 62,  64,  47,  151, 62,  251, 20,  36,
            63,  194, 101, 54,  63,  138, 51,  5,   63,  54,  35,  64,  63,
            94,  59,  32,  63,  28,  161, 44,  63,  137, 77,  46,  63,  199,
            12,  35,  63,  121, 21,  90,  62,  100, 168, 243, 62,  209, 97,
            143, 62,  9,   228, 61,  63,  223, 255, 239, 62,  58,  33,  29,
            63,  197, 186, 3,   63,  145, 27,  72,  63};
    thrust::host_vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_CreateFloatImage(3, 1, ref, ConversionType::Weighted);
}

TEST(Image, CreateFloatImage_3_1_Equal) {
    // reference data used to validate the creation of the float image
    uint8_t raw_ref[] = {
            44,  241, 17,  63,  29,  96,  75,  63,  154, 112, 20,  63,  255,
            240, 3,   63,  180, 56,  4,   63,  139, 60,  58,  63,  115, 8,
            204, 62,  215, 59,  119, 62,  64,  47,  151, 62,  251, 20,  36,
            63,  194, 101, 54,  63,  138, 51,  5,   63,  54,  35,  64,  63,
            94,  59,  32,  63,  28,  161, 44,  63,  137, 77,  46,  63,  199,
            12,  35,  63,  121, 21,  90,  62,  100, 168, 243, 62,  209, 97,
            143, 62,  9,   228, 61,  63,  223, 255, 239, 62,  58,  33,  29,
            63,  197, 186, 3,   63,  145, 27,  72,  63};
    thrust::host_vector<uint8_t> ref;
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
    thrust::host_vector<uint8_t> ref;
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
    thrust::host_vector<uint8_t> ref;
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
    thrust::host_vector<uint8_t> ref;
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
    thrust::host_vector<uint8_t> ref;
    for (int i = 0; i < 100; ++i) ref.push_back(raw_ref[i]);

    TEST_CreateFloatImage(3, 4, ref, ConversionType::Equal);
}

TEST(Image, FlipVerticalImage) {
    // reference data used to validate the creation of the float image
    // clang-format off
    uint8_t raw_input[] = {
      0, 1, 2, 3, 4, 5,
      6, 7, 8, 9, 10, 11,
      12, 13, 14, 15, 16, 17
    };
    thrust::host_vector<uint8_t> input;
    for (int i = 0; i < 18; ++i) input.push_back(raw_input[i]);
    uint8_t raw_flipped[] = {
      12, 13, 14, 15, 16, 17,
      6, 7, 8, 9, 10, 11,
      0, 1, 2, 3, 4, 5,
    };
    thrust::host_vector<uint8_t> flipped;
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
    thrust::host_vector<uint8_t> input;
    for (int i = 0; i < 18; ++i) input.push_back(raw_input[i]);
    uint8_t raw_flipped[] = {
      5, 4, 3, 2, 1, 0,
      11, 10, 9, 8, 7, 6,
      17, 16, 15, 14, 13, 12
    };
    thrust::host_vector<uint8_t> flipped;
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
