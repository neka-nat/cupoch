#include "cupoch/registration/permutohedral.h"
#include "cupoch/utility/platform.h"

namespace cupoch {
namespace registration {

namespace {

template<int Dim>
struct compute_lattice_key_value_functor {
    compute_lattice_key_value_functor(const Eigen::Matrix<float, Dim, 1>* features,
                                      LatticeCoordKey<Dim>* keys,
                                      float* weights,
                                      const Eigen::Matrix<float, Dim, 1> sigma)
        : inv_sigma_(1.0f / sigma.array()), features_(features),
        keys_(keys), weights_(weights) {};
    const Eigen::Matrix<float, Dim, 1> inv_sigma_;
    const Eigen::Matrix<float, Dim, 1>* features_;
    LatticeCoordKey<Dim>* keys_;
    float* weights_;
    __device__ void operator() (size_t idx) {
        float scaled_feature[Dim];
        //Scale the feature
        for (int k = 0; k < Dim; k++) scaled_feature[k] = features_[idx][k] * inv_sigma_[k];
        //Compute the lattice
        CreateLatticeGrid<Dim>(scaled_feature,
                               &keys_[idx * (Dim + 1)],
                               &weights_[idx * (Dim + 2)], true);
        weights_[idx * (Dim + 2) + Dim + 1] = -1.0;
    }
};

template<int Dim>
struct expand_copy_functor {
    expand_copy_functor(const Eigen::Vector3f* src, Eigen::Vector3f* dst)
    : src_(src), dst_(dst) {};
    const Eigen::Vector3f* src_;
    Eigen::Vector3f* dst_;
    __device__ void operator() (size_t idx) {
        for (int k = 0; k < Dim + 1; k++) {
            dst_[idx * (Dim + 1) + k] = src_[idx];
        }
    }
};

template<int Dim>
struct compute_lattice_info_functor {
    __device__ LatticeInfo operator() (const thrust::tuple<float, Eigen::Vector3f>& x) {
        float w = thrust::get<0>(x);
        Eigen::Vector3f vtx = thrust::get<1>(x);
        return LatticeInfo(w, w * vtx, w * vtx.squaredNorm());
    }
};

template <int Dim>
__global__
void map_insert_kernel(const LatticeCoordKey<Dim>* keys,
                       const LatticeInfo* values,
                       size_t n,
                       typename Permutohedral<Dim>::MapType lattice_map) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    lattice_map.emplace(keys[idx], values[idx]);
}

template<int Dim>
__global__
void compute_target_kernel(const LatticeCoordKey<Dim>* lattice_keys, const float* lattice_weights,
                           typename Permutohedral<Dim>::MapType lattice_map,
                           int n,
                           Eigen::Vector3f* target_vertices,
                           float* weights,
                           float* m2,
                           float outlier_constant) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    LatticeInfo aggregated_value;
    for (int lattice_j_idx = 0; lattice_j_idx < Dim + 1; ++lattice_j_idx) {
        //Get the lattice and weight
        const auto lattice_j = lattice_keys[idx * (Dim + 1) + lattice_j_idx];
        const float weight_j = lattice_weights[idx * (Dim + 2) + lattice_j_idx];
        const auto itr = lattice_map.find(lattice_j);
        if (itr != lattice_map.cend()) {
            aggregated_value += weight_j * itr->second;
        }
    }
    if(aggregated_value.weight_ < 1e-2f) {
         aggregated_value *= 0.0;
    } else {
        float w = aggregated_value.weight_;
        aggregated_value *= 1.0 / w;
        aggregated_value.weight_ = w / (w + outlier_constant);
    }
    target_vertices[idx] = aggregated_value.vertex_;
    weights[idx] = aggregated_value.weight_;
    m2[idx] = aggregated_value.vTv_;
}

struct compute_sigma_vlue_functor {
    __device__
    thrust::tuple<float, float> operator()(const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, float, float>& xyw) const {
        const Eigen::Vector3f x = thrust::get<0>(xyw);
        const Eigen::Vector3f m1 = thrust::get<1>(xyw);
        const float w = thrust::get<2>(xyw);
        const float m2 = thrust::get<3>(xyw);
        float upper = w * (x.squaredNorm() - 2.0f * m1.dot(x) + m2);
        float divisor = w;
        return thrust::make_tuple(upper, divisor);
    }
};

}

template <int Dim>
Permutohedral<Dim>::~Permutohedral() { MapType::destroyDeviceObject(lattice_map_); }

template <int Dim>
void Permutohedral<Dim>::BuildLatticeIndexNoBlur(const utility::device_vector<Eigen::Matrix<float, Dim, 1>>& obs_feature,
                                                 const utility::device_vector<Eigen::Vector3f>& obs_vertex) {
    if (obs_feature.size() != obs_vertex.size()) {
        utility::LogError("[BuildLatticeIndexNoBlur] Different array size between features and vertices.");
        return;
    }

    const size_t n = obs_feature.size();
    const size_t n_lt = n * (Dim + 1);
    utility::device_vector<LatticeCoordKey<Dim>> keys(n_lt);
    utility::device_vector<float> weights(n * (Dim + 2));
    utility::device_vector<Eigen::Vector3f> vertices(n_lt);
    compute_lattice_key_value_functor<Dim> func1(thrust::raw_pointer_cast(obs_feature.data()),
                                                 thrust::raw_pointer_cast(keys.data()),
                                                 thrust::raw_pointer_cast(weights.data()),
                                                 sigma_);
    expand_copy_functor<Dim> func2(thrust::raw_pointer_cast(obs_vertex.data()),
                                   thrust::raw_pointer_cast(vertices.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(n), func1);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(n), func2);
    auto w_end = thrust::remove_if(weights.begin(), weights.end(), [] __device__(float w) { return w < 0.0; });
    weights.resize(thrust::distance(weights.begin(), w_end));

    thrust::sort_by_key(keys.begin(), keys.end(), make_tuple_begin(weights, vertices),
                        [] __device__ (const LatticeCoordKey<Dim>& lhs, const LatticeCoordKey<Dim>& rhs) {
                            return lhs.less_than(rhs) < 0;
                        });
    utility::device_vector<LatticeCoordKey<Dim>> out_keys(n_lt);
    utility::device_vector<LatticeInfo> out_values(n_lt);
    compute_lattice_info_functor<Dim> info_fn;
    auto end = thrust::reduce_by_key(keys.begin(), keys.end(),
                                     thrust::make_transform_iterator(make_tuple_begin(weights, vertices), info_fn),
                                     out_keys.begin(), out_values.begin(),
                                     [] __device__ (const LatticeCoordKey<Dim>& lhs, const LatticeCoordKey<Dim>& rhs) {
                                         return lhs == rhs;
                                     },
                                     thrust::plus<LatticeInfo>());
    resize_all(thrust::distance(out_values.begin(), end.second), out_keys, out_values);
    lattice_map_ = MapType::createDeviceObject(n_lt);
    const dim3 threads(32);
    const dim3 blocks((n_lt / 2 + threads.x - 1) / threads.x);
    map_insert_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(out_keys.data()),
                                           thrust::raw_pointer_cast(out_values.data()), n_lt,
                                           lattice_map_);
    cudaSafeCall(cudaDeviceSynchronize());
}

template<int Dim>
void Permutohedral<Dim>::ComputeTarget(const utility::device_vector<Eigen::Matrix<float, Dim, 1>>& model_feature,
                                       utility::device_vector<Eigen::Vector3f>& target_vertices,
                                       utility::device_vector<float>& weights,
                                       utility::device_vector<float>& m2) {
    if (model_feature.size() != target_vertices.size() || model_feature.size() != weights.size()) {
        utility::LogError("[Premutohedral] Invalid device vector size.");
        return;
    }
    const int n = model_feature.size();
    utility::device_vector<LatticeCoordKey<Dim>> keys(n * (Dim + 1));
    utility::device_vector<float> w(n * (Dim + 2));
    compute_lattice_key_value_functor<Dim> func(thrust::raw_pointer_cast(model_feature.data()),
                                                thrust::raw_pointer_cast(keys.data()),
                                                thrust::raw_pointer_cast(w.data()),
                                                sigma_);
    thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(n), func);

    const dim3 threads(32);
    const dim3 blocks((n / 2 + threads.x - 1) / threads.x);
    compute_target_kernel<Dim><<<blocks, threads>>>(thrust::raw_pointer_cast(keys.data()),
                                                    thrust::raw_pointer_cast(w.data()),
                                                    lattice_map_, n,
                                                    thrust::raw_pointer_cast(target_vertices.data()),
                                                    thrust::raw_pointer_cast(weights.data()),
                                                    thrust::raw_pointer_cast(m2.data()),
                                                    outlier_constant_);
    cudaSafeCall(cudaDeviceSynchronize());
}

template<int Dim>
float Permutohedral<Dim>::ComputeSigma(const utility::device_vector<Eigen::Vector3f>& model,
                                       const utility::device_vector<Eigen::Vector3f>& target,
                                       const utility::device_vector<float>& weights,
                                       const utility::device_vector<float>& m2) {
    if (m2.size() != target.size() || m2.size() != model.size()) {
        utility::LogError("[Premutohedral] Invalid device vector size.");
        return 0.0f;
    }

    compute_sigma_vlue_functor func_tf;
    thrust::tuple<float, float> ud = thrust::transform_reduce(make_tuple_begin(model, target, weights, m2),
                                                              make_tuple_end(model, target, weights, m2),
                                                              func_tf, thrust::make_tuple(0.0f, 0.0f),
                                                              add_tuple_functor<float, float>());
    return std::sqrt(thrust::get<0>(ud) / (std::max(thrust::get<1>(ud), static_cast<float>(1.0e-6)) * 3.0f));
}

}
}