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
#include "tests/odometry/odometry_tools.h"

using namespace cupoch;
using namespace std;
using namespace unit_test;

shared_ptr<geometry::Image> odometry_tools::GenerateImage(
        const int& width,
        const int& height,
        const int& num_of_channels,
        const int& bytes_per_channel,
        const float& vmin,
        const float& vmax,
        const int& seed) {
    shared_ptr<geometry::Image> image = make_shared<geometry::Image>();

    image->Prepare(width, height, num_of_channels, bytes_per_channel);

    std::vector<uint8_t> data(image->data_.size());
    float* const depthData = Cast<float>(&data[0]);
    Rand(depthData, width * height, vmin, vmax, seed);
    image->SetData(data);

    return image;
}

// ----------------------------------------------------------------------------
// Shift the pixels left with a specified step.
// ----------------------------------------------------------------------------
void odometry_tools::ShiftLeft(shared_ptr<geometry::Image> image,
                               const int& step) {
    int width = image->width_;
    int height = image->height_;
    // int num_of_channels = image->num_of_channels_;
    // int bytes_per_channel = image->bytes_per_channel_;

    std::vector<uint8_t> data = image->GetData();
    float* const float_data = Cast<float>(&data[0]);
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            float_data[h * width + w] =
                    float_data[h * width + (w + step) % width];
    image->SetData(data);
}

// ----------------------------------------------------------------------------
// Shift the pixels up with a specified step.
// ----------------------------------------------------------------------------
void odometry_tools::ShiftUp(shared_ptr<geometry::Image> image,
                             const int& step) {
    int width = image->width_;
    int height = image->height_;
    // int num_of_channels = image->num_of_channels_;
    // int bytes_per_channel = image->bytes_per_channel_;

    std::vector<uint8_t> data = image->GetData();
    float* const float_data = Cast<float>(&data[0]);
    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++)
            float_data[h * width + w] =
                    float_data[((h + step) % height) * width + w];
    image->SetData(data);
}

// ----------------------------------------------------------------------------
// Create dummy correspondence map object.
// ----------------------------------------------------------------------------
shared_ptr<geometry::Image> odometry_tools::CorrespondenceMap(const int& width,
                                                              const int& height,
                                                              const int& vmin,
                                                              const int& vmax,
                                                              const int& seed) {
    int num_of_channels = 2;
    int bytes_per_channel = 4;

    shared_ptr<geometry::Image> image = make_shared<geometry::Image>();

    image->Prepare(width, height, num_of_channels, bytes_per_channel);

    std::vector<uint8_t> data(image->data_.size());
    int* const int_data = Cast<int>(&data[0]);
    size_t image_size = image->data_.size() / sizeof(int);
    Rand(int_data, image_size, vmin, vmax, seed);
    image->SetData(data);

    return image;
}

// ----------------------------------------------------------------------------
// Create dummy depth buffer object.
// ----------------------------------------------------------------------------
shared_ptr<geometry::Image> odometry_tools::DepthBuffer(const int& width,
                                                        const int& height,
                                                        const float& vmin,
                                                        const float& vmax,
                                                        const int& seed) {
    int num_of_channels = 1;
    int bytes_per_channel = 4;

    shared_ptr<geometry::Image> image = make_shared<geometry::Image>();

    image->Prepare(width, height, num_of_channels, bytes_per_channel);

    std::vector<uint8_t> data(image->data_.size());
    float* const float_data = Cast<float>(&data[0]);
    size_t image_size = image->data_.size() / sizeof(float);
    Rand(float_data, image_size, vmin, vmax, seed);
    image->SetData(data);

    return image;
}