#include "cupoc/utility/shared_ptr.hpp"
#include <cuda.h>

using namespace cupoc;
using namespace cupoc::utility;

__host__ __device__
sp_counter_impl_p::sp_counter_impl_p() : sp_counter_base() {}

__host__
void sp_counter_impl_p::dispose(void* p) {if(p) cudaFree(reinterpret_cast<void*>(p));}
