#pragma once

#include <mutex>
#include <random>

#include "cupoch/utility/console.h"

namespace cupoch {
namespace utility {
namespace random {

/// Set Open3D global random seed.
void Seed(const int seed);

/// Get global singleton random engine.
/// You must also lock the global mutex before calling the engine.
///
/// Example:
/// ```cpp
/// #include "open3d/utility/Random.h"
///
/// {
///     // Put the lock and the call to the engine in the same scope.
///     std::lock_guard<std::mutex> lock(*utility::random::GetMutex());
///     std::shuffle(vals.begin(), vals.end(), *utility::random::GetEngine());
/// }
/// ```
std::mt19937* GetEngine();

/// Get global singleton mutex to protect the engine call. Also see
/// random::GetEngine().
std::mutex* GetMutex();

/// Generate a random uint32.
/// This function is globally seeded by utility::random::Seed().
/// This function is automatically protected by the global random mutex.
uint32_t RandUint32();

/// Generate uniformly distributed random integers in [low, high).
/// This class is globally seeded by utility::random::Seed().
/// This class is a wrapper around std::uniform_int_distribution.
///
/// Example:
/// ```cpp
/// #include "open3d/utility/Random.h"
///
/// // Globally seed Open3D. This will affect all random functions.
/// utility::random::Seed(0);
///
/// // Generate a random int in [0, 100).
/// utility::random::UniformIntGenerator<int> gen(0, 100);
/// for (size_t i = 0; i < 10; i++) {
///     std::cout << gen() << std::endl;
/// }
/// ```
template <typename T>
class UniformIntGenerator {
public:
    /// Generate uniformly distributed random integer from
    /// [low, low + 1, ... high - 1].
    ///
    /// \param low The lower bound (inclusive).
    /// \param high The upper bound (exclusive). \p high must be > \p low.
    UniformIntGenerator(const T low, const T high) : distribution_(low, high) {
        if (low < 0) {
            utility::LogError("low must be > 0, but got {}.", low);
        }
        if (low >= high) {
            utility::LogError("low must be < high, but got low={} and high={}.",
                              low, high);
        }
    }

    /// Call this to generate a uniformly distributed integer.
    T operator()() {
        std::lock_guard<std::mutex> lock(*GetMutex());
        return distribution_(*GetEngine());
    }

protected:
    std::uniform_int_distribution<T> distribution_;
};

}
}  // namespace utility
}  // namespace cupoch
