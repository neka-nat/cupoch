#pragma once
#include <cuda_runtime.h>

namespace cupoch {
namespace utility {

static const size_t MAX_NUM_STREAMS = 16;

cudaStream_t GetStream(size_t i);

int GetDevice();

void SetDevice(int device_no);

}
}