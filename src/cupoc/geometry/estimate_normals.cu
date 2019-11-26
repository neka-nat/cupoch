#include <Eigen/Geometry>
#include "cupoc/geometry/pointcloud.h"
#include "cupoc/geometry/kdtree_flann.h"
#include "cupoc/utility/console.h"

using namespace cupoc;
using namespace cupoc::geometry;

namespace {

__device__
Eigen::Vector3f ComputeEigenvector0(const Eigen::Matrix3f &A, float eval0) {
    Eigen::Vector3f row0(A(0, 0) - eval0, A(0, 1), A(0, 2));
    Eigen::Vector3f row1(A(0, 1), A(1, 1) - eval0, A(1, 2));
    Eigen::Vector3f row2(A(0, 2), A(1, 2), A(2, 2) - eval0);
    Eigen::Vector3f r0xr1 = row0.cross(row1);
    Eigen::Vector3f r0xr2 = row0.cross(row2);
    Eigen::Vector3f r1xr2 = row1.cross(row2);
    float d0 = r0xr1.dot(r0xr1);
    float d1 = r0xr2.dot(r0xr2);
    float d2 = r1xr2.dot(r1xr2);

    float dmax = d0;
    int imax = 0;
    if (d1 > dmax) {
        dmax = d1;
        imax = 1;
    }
    if (d2 > dmax) {
        imax = 2;
    }

    if (imax == 0) {
        return r0xr1 / std::sqrt(d0);
    } else if (imax == 1) {
        return r0xr2 / std::sqrt(d1);
    } else {
        return r1xr2 / std::sqrt(d2);
    }
}
    
__device__
Eigen::Vector3f ComputeEigenvector1(const Eigen::Matrix3f &A,
                                    const Eigen::Vector3f &evec0,
                                    double eval1) {
    Eigen::Vector3f U, V;
    if (std::abs(evec0(0)) > std::abs(evec0(1))) {
        float inv_length =
                1 / std::sqrt(evec0(0) * evec0(0) + evec0(2) * evec0(2));
        U << -evec0(2) * inv_length, 0, evec0(0) * inv_length;
    } else {
        float inv_length =
                1 / std::sqrt(evec0(1) * evec0(1) + evec0(2) * evec0(2));
        U << 0, evec0(2) * inv_length, -evec0(1) * inv_length;
    }
    V = evec0.cross(U);

    Eigen::Vector3f AU(A(0, 0) * U(0) + A(0, 1) * U(1) + A(0, 2) * U(2),
                      A(0, 1) * U(0) + A(1, 1) * U(1) + A(1, 2) * U(2),
                      A(0, 2) * U(0) + A(1, 2) * U(1) + A(2, 2) * U(2));

    Eigen::Vector3f AV = {A(0, 0) * V(0) + A(0, 1) * V(1) + A(0, 2) * V(2),
                          A(0, 1) * V(0) + A(1, 1) * V(1) + A(1, 2) * V(2),
                          A(0, 2) * V(0) + A(1, 2) * V(1) + A(2, 2) * V(2)};
    
    float m00 = U(0) * AU(0) + U(1) * AU(1) + U(2) * AU(2) - eval1;
    float m01 = U(0) * AV(0) + U(1) * AV(1) + U(2) * AV(2);
    float m11 = V(0) * AV(0) + V(1) * AV(1) + V(2) * AV(2) - eval1;
    
    float absM00 = std::abs(m00);
    float absM01 = std::abs(m01);
    float absM11 = std::abs(m11);
    float max_abs_comp;
    if (absM00 >= absM11) {
        max_abs_comp = max(absM00, absM01);
        if (max_abs_comp > 0) {
            if (absM00 >= absM01) {
                m01 /= m00;
                m00 = 1 / std::sqrt(1 + m01 * m01);
                m01 *= m00;
            } else {
                m00 /= m01;
                m01 = 1 / std::sqrt(1 + m00 * m00);
                m00 *= m01;
            }
            return m01 * U - m00 * V;
        } else {
            return U;
        }
    } else {
        max_abs_comp = max(absM11, absM01);
        if (max_abs_comp > 0) {
            if (absM11 >= absM01) {
                m01 /= m11;
                m11 = 1 / std::sqrt(1 + m01 * m01);
                m01 *= m11;
            } else {
                m11 /= m01;
                m01 = 1 / std::sqrt(1 + m11 * m11);
                m11 *= m01;
            }
            return m11 * U - m01 * V;
        } else {
            return U;
        }
    }
}

__device__
Eigen::Vector3f FastEigen3x3(Eigen::Matrix3f &A) {
    // Previous version based on:
    // https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
    // Current version based on
    // https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    // which handles edge cases like points on a plane

    float max_coeff = A.maxCoeff();
    if (max_coeff == 0) {
        return Eigen::Vector3f::Zero();
    }
    A /= max_coeff;

    float norm = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
    if (norm > 0) {
        Eigen::Vector3f eval;
        Eigen::Vector3f evec0;
        Eigen::Vector3f evec1;
        Eigen::Vector3f evec2;

        float q = (A(0, 0) + A(1, 1) + A(2, 2)) / 3;

        float b00 = A(0, 0) - q;
        float b11 = A(1, 1) - q;
        float b22 = A(2, 2) - q;

        float p =
                std::sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);

        float c00 = b11 * b22 - A(1, 2) * A(1, 2);
        float c01 = A(0, 1) * b22 - A(1, 2) * A(0, 2);
        float c02 = A(0, 1) * A(1, 2) - b11 * A(0, 2);
        float det = (b00 * c00 - A(0, 1) * c01 + A(0, 2) * c02) / (p * p * p);

        float half_det = det * 0.5;
        half_det = min(max(half_det, -1.0), 1.0);

        float angle = std::acos(half_det) / (float)3;
        float const two_thirds_pi = 2.09439510239319549;
        float beta2 = std::cos(angle) * 2;
        float beta0 = std::cos(angle + two_thirds_pi) * 2;
        float beta1 = -(beta0 + beta2);

        eval(0) = q + p * beta0;
        eval(1) = q + p * beta1;
        eval(2) = q + p * beta2;

        if (half_det >= 0) {
            evec2 = ComputeEigenvector0(A, eval(2));
            if (eval(2) < eval(0) && eval(2) < eval(1)) {
                A *= max_coeff;
                return evec2;
            }
            evec1 = ComputeEigenvector1(A, evec2, eval(1));
            A *= max_coeff;
            if (eval(1) < eval(0) && eval(1) < eval(2)) {
                return evec1;
            }
            evec0 = evec1.cross(evec2);
            return evec0;
        } else {
            evec0 = ComputeEigenvector0(A, eval(0));
            if (eval(0) < eval(1) && eval(0) < eval(2)) {
                A *= max_coeff;
                return evec0;
            }
            evec1 = ComputeEigenvector1(A, evec0, eval(1));
            A *= max_coeff;
            if (eval(1) < eval(0) && eval(1) < eval(2)) {
                return evec1;
            }
            evec2 = evec0.cross(evec1);
            return evec2;
        }
    } else {
        A *= max_coeff;
        if (A(0, 0) < A(1, 1) && A(0, 0) < A(2, 2)) {
            return Eigen::Vector3f(1, 0, 0);
        } else if (A(1, 1) < A(0, 0) && A(1, 1) < A(2, 2)) {
            return Eigen::Vector3f(0, 1, 0);
        } else {
            return Eigen::Vector3f(0, 0, 1);
        }
    }
}

__device__
Eigen::Vector3f_u ComputeNormal(const Eigen::Vector3f_u* points,
                                const KNNIndices &indices, int knn) {
    if (indices[0] < 0) return Eigen::Vector3f_u(0.0, 0.0, 1.0);

    Eigen::Matrix3f covariance;
    Eigen::Matrix<float, 9, 1> cumulants;
    cumulants.setZero();
    for (size_t i = 0; i < knn; i++) {
        const Eigen::Vector3f_u& point = points[indices[i]];
        cumulants(0) += point(0);
        cumulants(1) += point(1);
        cumulants(2) += point(2);
        cumulants(3) += point(0) * point(0);
        cumulants(4) += point(0) * point(1);
        cumulants(5) += point(0) * point(2);
        cumulants(6) += point(1) * point(1);
        cumulants(7) += point(1) * point(2);
        cumulants(8) += point(2) * point(2);
    }
    cumulants /= (double)indices.size();
    covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
    covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
    covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
    covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
    covariance(1, 0) = covariance(0, 1);
    covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
    covariance(2, 0) = covariance(0, 2);
    covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
    covariance(2, 1) = covariance(1, 2);

    return FastEigen3x3(covariance);
}

struct compute_normal_functor {
    compute_normal_functor(const Eigen::Vector3f_u* points,
                           const int* indices, int knn)
        : points_(points), indices_(indices), knn_(knn) {};
    const Eigen::Vector3f_u* points_;
    const int* indices_;
    const int knn_;
    __device__
    Eigen::Vector3f_u operator()(const int& idx) const {
        KNNIndices idxs = KNNIndices::Constant(-1);
        for (int k = 0; k < knn_; ++k) idxs[k] = indices_[idx * knn_ + k];
        Eigen::Vector3f_u normal = ComputeNormal(points_, idxs, knn_);
        if (normal.norm() == 0.0) {
            normal = Eigen::Vector3f_u(0.0, 0.0, 1.0);
        }
        return normal;
    }
};

struct align_normals_direction {
    align_normals_direction(const Eigen::Vector3f& orientation_reference)
        : orientation_reference_(orientation_reference) {};
    const Eigen::Vector3f orientation_reference_;
    __device__
    void operator()(Eigen::Vector3f_u& normal) const {
        if (normal.norm() == 0.0) {
            normal = orientation_reference_;
        } else if (normal.dot(orientation_reference_) < 0.0) {
            normal *= -1.0;
        }
    }
};

}

bool PointCloud::EstimateNormals(const KDTreeSearchParam &search_param) {
    const bool has_normal = HasNormals();
    if (HasNormals() == false) {
        normals_.resize(points_.size());
    }
    KDTreeFlann kdtree;
    kdtree.SetGeometry(*this);
    thrust::device_vector<int> indices;
    thrust::device_vector<float> distance2;
    kdtree.Search(points_, search_param, indices, distance2);
    normals_.resize(points_.size());
    compute_normal_functor func(thrust::raw_pointer_cast(points_.data()),
                                thrust::raw_pointer_cast(indices.data()),
                                ((const KDTreeSearchParamKNN &)search_param).knn_);
    thrust::transform(indices.begin(), indices.end(), normals_.begin(), func);
    return true;
}

bool PointCloud::OrientNormalsToAlignWithDirection(const Eigen::Vector3f &orientation_reference) {
    if (HasNormals() == false) {
        utility::LogWarning(
                "[OrientNormalsToAlignWithDirection] No normals in the "
                "PointCloud. Call EstimateNormals() first.\n");
        return false;
    }
    align_normals_direction func(orientation_reference);
    thrust::for_each(normals_.begin(), normals_.end(), func);
    return true;
}
