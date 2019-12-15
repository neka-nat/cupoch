#pragma once
#include <cuda_runtime.h>

namespace cupoch {
namespace utility {

static const size_t MAX_DEVICES = 16;

cudaStream_t GetGlobalStream(size_t i);

}
}