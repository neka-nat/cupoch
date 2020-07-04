#include "cupoch/registration/lattice_utils.h"

namespace cupoch {
namespace registration {

namespace {

template <int Dim>
__host__ __device__ float Scale(int index, bool no_blur) {
    return (Dim + 1) * sqrtf(((no_blur) ? (1.0f / 6.0f) : (2.0f / 3.0f)) /
                             ((index + 1) * (index + 2)));
}

}  // namespace

// The method to create the lattice grid from lattice
template <int Dim>
void CreateLatticeGrid(float *feature,
                       LatticeCoordKey<Dim> *lattice_coord_keys,
                       float *barycentric,
                       bool no_blur) {
    float elevated[Dim + 1];
    elevated[Dim] = -Dim * (feature[Dim - 1]) * Scale<Dim>(Dim - 1, no_blur);
#pragma unroll
    for (int i = Dim - 1; i > 0; --i) {
        elevated[i] = (elevated[i + 1] -
                       i * (feature[i - 1]) * Scale<Dim>(i - 1, no_blur) +
                       (i + 2) * (feature[i]) * Scale<Dim>(i, no_blur));
    }
    elevated[0] = elevated[1] + 2 * (feature[0]) * Scale<Dim>(0, no_blur);

    short greedy[Dim + 1];
    short sum = 0;
#pragma unroll
    for (int i = 0; i <= Dim; ++i) {
        float v = elevated[i] * (1.0f / (Dim + 1));
        float up = ceilf(v) * (Dim + 1);
        float down = floorf(v) * (Dim + 1);
        greedy[i] = static_cast<short>(
                (up - elevated[i] < elevated[i] - down) ? up : down);
        sum += greedy[i];
    }
    sum /= Dim + 1;

    // Sort differential to find the permutation between this simplex and the
    // canonical one
    short rank[Dim + 1] = {0};
    if (no_blur) {
#pragma unroll
        for (int i = 0; i < Dim; ++i) {
#pragma unroll
            for (int j = i + 1; j <= Dim; ++j) {
                ++rank[(elevated[i] - greedy[i] < elevated[j] - greedy[j]) ? i
                                                                           : j];
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < Dim; ++i) {
#pragma unroll
            for (int j = 0; j <= Dim; ++j) {
                if (elevated[i] - greedy[i] < elevated[j] - greedy[j] ||
                    (elevated[i] - greedy[i] == elevated[j] - greedy[j] &&
                     i > j))
                    ++rank[i];
            }
        }
    }

    // Sum too large, need to bring down the ones with the smallest differential
    if (sum > 0) {
#pragma unroll
        for (int i = 0; i <= Dim; ++i) {
            if (rank[i] >= Dim + 1 - sum) {
                greedy[i] -= Dim + 1;
                rank[i] += sum - (Dim + 1);
            } else {
                rank[i] += sum;
            }
        }
    } else if (sum < 0) {  // Sum too small, need to bring up the ones with
                           // largest differential
#pragma unroll
        for (int i = 0; i <= Dim; ++i) {
            if (rank[i] < -sum) {
                greedy[i] += Dim + 1;
                rank[i] += (Dim + 1) + sum;
            } else {
                rank[i] += sum;
            }
        }
    }

// Turn delta into barycentric coords
#pragma unroll
    for (int i = 0; i <= Dim + 1; ++i) {
        barycentric[i] = 0;
    }

#pragma unroll
    for (int i = 0; i <= Dim; ++i) {
        float delta = (elevated[i] - greedy[i]) * (1.0f / (Dim + 1));
        barycentric[Dim - rank[i]] += delta;
        barycentric[Dim + 1 - rank[i]] -= delta;
    }
    barycentric[0] += 1.0f + barycentric[Dim + 1];

// Construct the key and their weight
#pragma unroll
    for (int color = 0; color <= Dim; ++color) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        Eigen::Matrix<short, Dim, 1> &key = lattice_coord_keys[color].key_;
        for (int i = 0; i < Dim; ++i) {
            key[i] = greedy[i] + color;
            if (rank[i] > Dim - color) key[i] -= (Dim + 1);
        }
    }
}

}  // namespace registration
}  // namespace cupoch