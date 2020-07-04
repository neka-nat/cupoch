#pragma once

#include <Eigen/Core>
#include <stdgpu/unordered_map.cuh>

#include "cupoch/registration/lattice_utils.h"
#include "cupoch/utility/device_vector.h"

namespace cupoch {
namespace registration {

struct LatticeInfo {
    __host__ __device__
    LatticeInfo(float weight = 0,
                const Eigen::Vector3f& vertex = Eigen::Vector3f::Zero(),
                float vTv = 0,
                const Eigen::Vector3f& normal = Eigen::Vector3f::Zero())
        : weight_(weight), vertex_(vertex), vTv_(vTv), normal_(normal){};
    __host__ __device__ ~LatticeInfo(){};
    __host__ __device__ LatticeInfo(const LatticeInfo& other)
        : weight_(other.weight_),
          vertex_(other.vertex_),
          vTv_(other.vTv_),
          normal_(other.normal_){};

    __host__ __device__ LatticeInfo& operator+=(const LatticeInfo& other) {
        weight_ += other.weight_;
        vertex_ += other.vertex_;
        vTv_ += other.vTv_;
        normal_ += other.normal_;
        return *this;
    }

    __host__ __device__ LatticeInfo& operator*=(float other) {
        weight_ *= other;
        vertex_ *= other;
        vTv_ *= other;
        normal_ *= other;
        return *this;
    }

    float weight_;
    Eigen::Vector3f vertex_;
    float vTv_;
    Eigen::Vector3f normal_;
};

__host__ __device__ inline LatticeInfo operator+(const LatticeInfo& rhs,
                                                 const LatticeInfo& lhs) {
    LatticeInfo ans = rhs;
    ans += lhs;
    return ans;
}

__host__ __device__ inline LatticeInfo operator*(float rhs,
                                                 const LatticeInfo& lhs) {
    LatticeInfo ans = lhs;
    ans *= rhs;
    return ans;
}

template <int Dim>
class Permutohedral {
public:
    // The hash and index for permutohedral lattice
    struct PermutohedralHasher {
        __host__ __device__ size_t
        operator()(const LatticeCoordKey<Dim>& lattice) const {
            return lattice.hash();
        }
    };
    typedef stdgpu::unordered_map<LatticeCoordKey<Dim>,
                                  LatticeInfo,
                                  PermutohedralHasher>
            MapType;

    Permutohedral(float sigma) : sigma_(Eigen::Vector3f::Constant(sigma)){};
    ~Permutohedral();

    void BuildLatticeIndexNoBlur(
            const utility::device_vector<Eigen::Matrix<float, Dim, 1>>&
                    obs_feature,
            const utility::device_vector<Eigen::Vector3f>& obs_vertex);

    void ComputeTarget(const utility::device_vector<
                               Eigen::Matrix<float, Dim, 1>>& model_feature,
                       utility::device_vector<Eigen::Vector3f>& target_vertices,
                       utility::device_vector<float>& weights,
                       utility::device_vector<float>& m2);

    float ComputeSigma(const utility::device_vector<Eigen::Vector3f>& model,
                       const utility::device_vector<Eigen::Vector3f>& target,
                       const utility::device_vector<float>& weights,
                       const utility::device_vector<float>& m2);

public:
    MapType lattice_map_;
    Eigen::Matrix<float, Dim, 1> sigma_;
    float outlier_constant_ = 0.2;
};

}  // namespace registration
}  // namespace cupoch

#include "cupoch/registration/permutohedral.inl"