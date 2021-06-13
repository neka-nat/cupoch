/**
 * Copyright (c) 2021 Neka-Nat
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
#include "cupoch/utility/eigenvalue.h"

namespace cupoch {
namespace utility {

__device__ float signf(float x) { return x / fabs(x); }

__device__ Eigen::Vector3f ComputeEigenvector0(const Eigen::Matrix3f &A,
                                               float eval0) {
    Eigen::Vector3f row0(A(0, 0) - eval0, A(0, 1), A(0, 2));
    Eigen::Vector3f row1(A(0, 1), A(1, 1) - eval0, A(1, 2));
    Eigen::Vector3f row2(A(0, 2), A(1, 2), A(2, 2) - eval0);
    Eigen::Vector3f rxr[3];
    rxr[0] = row0.cross(row1);
    rxr[1] = row0.cross(row2);
    rxr[2] = row1.cross(row2);
    Eigen::Vector3f d;
    d[0] = rxr[0].dot(rxr[0]);
    d[1] = rxr[1].dot(rxr[1]);
    d[2] = rxr[2].dot(rxr[2]);

    int imax;
    d.maxCoeff(&imax);
    return rxr[imax] / sqrtf(d[imax]);
}

__device__ Eigen::Vector3f ComputeEigenvector1(const Eigen::Matrix3f &A,
                                               const Eigen::Vector3f &evec0,
                                               float eval1) {
    float max_evec0_abs = max(fabs(evec0(0)), fabs(evec0(1)));
    float inv_length =
            1 / sqrtf(max_evec0_abs * max_evec0_abs + evec0(2) * evec0(2));
    Eigen::Vector3f U = (fabs(evec0(0)) > fabs(evec0(1)))
                                ? Eigen::Vector3f(-evec0(2), 0, evec0(0))
                                : Eigen::Vector3f(0, evec0(2), -evec0(1));
    U *= inv_length;
    Eigen::Vector3f V = evec0.cross(U);

    Eigen::Vector3f AU(A(0, 0) * U(0) + A(0, 1) * U(1) + A(0, 2) * U(2),
                       A(0, 1) * U(0) + A(1, 1) * U(1) + A(1, 2) * U(2),
                       A(0, 2) * U(0) + A(1, 2) * U(1) + A(2, 2) * U(2));

    Eigen::Vector3f AV = {A(0, 0) * V(0) + A(0, 1) * V(1) + A(0, 2) * V(2),
                          A(0, 1) * V(0) + A(1, 1) * V(1) + A(1, 2) * V(2),
                          A(0, 2) * V(0) + A(1, 2) * V(1) + A(2, 2) * V(2)};

    float m00 = U(0) * AU(0) + U(1) * AU(1) + U(2) * AU(2) - eval1;
    float m01 = U(0) * AV(0) + U(1) * AV(1) + U(2) * AV(2);
    float m11 = V(0) * AV(0) + V(1) * AV(1) + V(2) * AV(2) - eval1;

    float absM00 = fabs(m00);
    float absM01 = fabs(m01);
    float absM11 = fabs(m11);
    float max_abs_comp0 = max(absM00, absM11);
    float max_abs_comp = max(max_abs_comp0, absM01);
    float coef2 = min(max_abs_comp0, absM01) / max(max_abs_comp, 1.0e-6);
    float coef1 = 1.0 / sqrtf(1.0 + coef2 * coef2);
    if (absM00 >= absM11) {
        coef2 *= coef1 * signf(m00) * signf(m01);
        return (max_abs_comp0 >= absM01) ? coef2 * U - coef1 * V
                                         : coef1 * U - coef2 * V;
    } else {
        coef2 *= coef1 * signf(m11) * signf(m01);
        return (max_abs_comp0 >= absM01) ? coef1 * U - coef2 * V
                                         : coef2 * U - coef1 * V;
    }
}

__device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector3f> FastEigen3x3(Eigen::Matrix3f &A) {
    // Previous version based on:
    // https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
    // Current version based on
    // https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    // which handles edge cases like points on a plane

    float max_coeff = A.maxCoeff();
    if (max_coeff == 0) {
        return thrust::make_tuple(Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero());
    }
    A /= max_coeff;

    float norm = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
    if (norm > 0) {
        Eigen::Vector3f eval;
        Eigen::Matrix3f evec;

        float q = (A(0, 0) + A(1, 1) + A(2, 2)) / 3;

        float b00 = A(0, 0) - q;
        float b11 = A(1, 1) - q;
        float b22 = A(2, 2) - q;

        float p = sqrtf((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);

        float c00 = b11 * b22 - A(1, 2) * A(1, 2);
        float c01 = A(0, 1) * b22 - A(1, 2) * A(0, 2);
        float c02 = A(0, 1) * A(1, 2) - b11 * A(0, 2);
        float det = (b00 * c00 - A(0, 1) * c01 + A(0, 2) * c02) / (p * p * p);

        float half_det = det * 0.5;
        half_det = min(max(half_det, -1.0), 1.0);

        float angle = acos(half_det) / (float)3;
        float const two_thirds_pi = 2.09439510239319549;
        float beta2 = cos(angle) * 2;
        float beta0 = cos(angle + two_thirds_pi) * 2;
        float beta1 = -(beta0 + beta2);

        eval(0) = q + p * beta0;
        eval(1) = q + p * beta1;
        eval(2) = q + p * beta2;

        if (half_det >= 0) {
            evec.col(2) = ComputeEigenvector0(A, eval(2));
            evec.col(1) = ComputeEigenvector1(A, evec.col(2), eval(1));
            evec.col(0) = evec.col(1).cross(evec.col(2));
            int min_id, max_id;
            eval.minCoeff(&min_id);
            eval.maxCoeff(&max_id);
            A *= max_coeff;
            return thrust::make_tuple(evec.col(min_id), evec.col(max_id));
        } else {
            evec.col(0) = ComputeEigenvector0(A, eval(0));
            evec.col(1) = ComputeEigenvector1(A, evec.col(0), eval(1));
            evec.col(2) = evec.col(0).cross(evec.col(1));
            A *= max_coeff;
            int min_id, max_id;
            eval.minCoeff(&min_id);
            eval.maxCoeff(&max_id);
            A *= max_coeff;
            return thrust::make_tuple(evec.col(min_id), evec.col(max_id));
        }
    } else {
        A *= max_coeff;
        int min_id, max_id;
        A.diagonal().minCoeff(&min_id);
        A.diagonal().maxCoeff(&max_id);
        return thrust::make_tuple(Eigen::Vector3f::Unit(min_id), Eigen::Vector3f::Unit(max_id));
    }
}

}
}