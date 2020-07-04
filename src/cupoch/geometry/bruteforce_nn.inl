#include "cupoch/geometry/bruteforce_nn.h"
#include "cupoch/utility/platform.h"

#define THREAD_2D_UNIT 16
#define DIV_CEILING(a, b) ((a + b - 1) / b)

namespace cupoch {
namespace geometry {

namespace {

template <int Dim>
__global__ void ComputeDistancesKernel(
        const Eigen::Matrix<float, Dim, 1>* ref,
        const Eigen::Matrix<float, Dim, 1>* query,
        float* distance_mat,
        int ref_size,
        int query_size) {
    __shared__ float shared_query[THREAD_2D_UNIT][THREAD_2D_UNIT];
    __shared__ float shared_ref[THREAD_2D_UNIT][THREAD_2D_UNIT];

    const int tx = threadIdx.x, ty = threadIdx.y;
    int query_base = blockIdx.x * blockDim.x;
    int ref_base = blockIdx.y * blockDim.y;

    int query_idx = query_base + tx;
    int ref_idx_local = ref_base + tx;
    int ref_idx_global = ref_base + ty;

    bool mask_query = query_idx < query_size;
    bool mask_ref_local = ref_idx_local < ref_size;
    bool mask_ref_global = ref_idx_global < ref_size;

    float ssd = 0.0;
    for (int feature_batch = 0; feature_batch < Dim;
         feature_batch += THREAD_2D_UNIT) {
        int feature_idx = feature_batch + ty;
        bool mask_feature = feature_idx < Dim;
        shared_query[ty][tx] = (mask_query && mask_feature)
                                       ? query[query_idx][feature_idx]
                                       : 0;
        shared_ref[ty][tx] = (mask_ref_local && mask_feature)
                                     ? ref[ref_idx_local][feature_idx]
                                     : 0;
        __syncthreads();

        /* Here ty denotes reference entry index */
        if (mask_query && mask_ref_global) {
            for (int j = 0; j < THREAD_2D_UNIT; ++j) {
                float diff = shared_query[j][tx] - shared_ref[j][ty];
                ssd += diff * diff;
            }
        }
        __syncthreads();
    }

    if (mask_query && mask_ref_global) {
        distance_mat[ref_idx_global * query_size + query_idx] = ssd;
    }
}

__global__ void FindNNKernel(const float* distance_mat,
                             int* indices,
                             float* distances,
                             int ref_size,
                             int query_size) {
    const int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= query_size) return;

    int nn_idx = 0;
    float nn_dist = distance_mat[0 * query_size + query_idx];
    for (int i = 1; i < ref_size; ++i) {
        float dist = distance_mat[i * query_size + query_idx];
        if (dist < nn_dist) {
            nn_dist = dist;
            nn_idx = i;
        }
    }

    indices[query_idx] = nn_idx;
    distances[query_idx] = nn_dist;
}

}  // namespace

template <int Dim>
void BruteForceNN(
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>>& ref,
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>>& query,
        utility::device_vector<int>& indices,
        utility::device_vector<float>& distances) {
    utility::device_vector<float> distance_mat(ref.size() * query.size());
    indices.resize(query.size());
    distances.resize(query.size());
    const dim3 blocks1(DIV_CEILING(query.size(), THREAD_2D_UNIT),
                       DIV_CEILING(ref.size(), THREAD_2D_UNIT));
    const dim3 threads1(THREAD_2D_UNIT, THREAD_2D_UNIT);
    ComputeDistancesKernel<<<blocks1, threads1>>>(
            thrust::raw_pointer_cast(ref.data()),
            thrust::raw_pointer_cast(query.data()),
            thrust::raw_pointer_cast(distance_mat.data()), ref.size(),
            query.size());
    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaGetLastError());

    const dim3 blocks2(DIV_CEILING(query.size(), 256));
    const dim3 threads2(256);
    FindNNKernel<<<blocks2, threads2>>>(
            thrust::raw_pointer_cast(distance_mat.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(distances.data()), ref.size(),
            query.size());
    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaGetLastError());
}

}  // namespace geometry
}  // namespace cupoch