#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/helper.h"
#include <thrust/gather.h>


using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct compute_key_functor {
    compute_key_functor(const Eigen::Vector3f& voxel_min_bound, float voxel_size)
        : voxel_min_bound_(voxel_min_bound), voxel_size_(voxel_size) {};
    const Eigen::Vector3f voxel_min_bound_;
    const float voxel_size_;
    __device__
    Eigen::Vector3i operator()(const Eigen::Vector3f_u& pt) {
        auto ref_coord = (pt - voxel_min_bound_) / voxel_size_;
        return Eigen::Vector3i(int(floor(ref_coord(0))), int(floor(ref_coord(1))), int(floor(ref_coord(2))));
    }
};

template<typename OutputIterator, class... Args>
__host__
int CalcAverageByKey(thrust::device_vector<Eigen::Vector3i>& keys,
                     OutputIterator buf_begins, OutputIterator output_begins) {
    const size_t n = keys.size();
    thrust::sort_by_key(keys.begin(), keys.end(), buf_begins);

    thrust::device_vector<Eigen::Vector3i> keys_out(n);
    thrust::device_vector<int> counts(n);
    auto end1 = thrust::reduce_by_key(keys.begin(), keys.end(),
                                      thrust::make_constant_iterator(1),
                                      keys_out.begin(), counts.begin());
    int n_out = thrust::distance(counts.begin(), end1.second);
    counts.resize(n_out);

    thrust::equal_to<Eigen::Vector3i> binary_pred;
    add_tuple_functor<Args...> add_func;
    auto end2 = thrust::reduce_by_key(keys.begin(), keys.end(), buf_begins,
                                      keys_out.begin(), output_begins,
                                      binary_pred, add_func);

    devided_tuple_functor<Args...> dv_func;
    thrust::transform(output_begins, output_begins + n_out,
                      counts.begin(), output_begins,
                      dv_func);
    return n_out;
}

}

std::shared_ptr<PointCloud> PointCloud::SelectDownSample(const thrust::device_vector<size_t> &indices, bool invert) const {
    auto output = std::make_shared<PointCloud>();
    const bool has_normals = HasNormals();
    const bool has_colors = HasColors();

    output->points_.resize(indices.size());
    thrust::gather(indices.begin(), indices.end(), points_.begin(), output->points_.begin());
    if (HasNormals()) {
        output->normals_.resize(indices.size());
        thrust::gather(indices.begin(), indices.end(), normals_.begin(), output->normals_.begin());
    }
    if (HasColors()) {
        output->colors_.resize(indices.size());
        thrust::gather(indices.begin(), indices.end(), colors_.begin(), output->colors_.begin());
    }
    return output;
}

std::shared_ptr<PointCloud> PointCloud::VoxelDownSample(float voxel_size) const {
    auto output = std::make_shared<PointCloud>();
    if (voxel_size <= 0.0) {
        utility::LogWarning("[VoxelDownSample] voxel_size <= 0.\n");
        return output;
    }

    const Eigen::Vector3f voxel_size3 = Eigen::Vector3f(voxel_size, voxel_size, voxel_size);
    const Eigen::Vector3f voxel_min_bound = GetMinBound() - voxel_size3 * 0.5;
    const Eigen::Vector3f voxel_max_bound = GetMaxBound() + voxel_size3 * 0.5;

    if (voxel_size * std::numeric_limits<int>::max() < (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        utility::LogWarning("[VoxelDownSample] voxel_size is too small.\n");
        return output;
    }

    const int n = points_.size();
    const bool has_normals = HasNormals();
    const bool has_colors = HasColors();
    compute_key_functor ck_func(voxel_min_bound, voxel_size);
    thrust::device_vector<Eigen::Vector3i> keys(n);
    thrust::transform(points_.begin(), points_.end(), keys.begin(), ck_func);

    thrust::device_vector<Eigen::Vector3f_u> sorted_points = points_;
    output->points_.resize(n);
    if (!has_normals && !has_colors) {
        typedef thrust::tuple<thrust::device_vector<Eigen::Vector3f_u>::iterator> IteratorTuple;
        typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
        auto n_out = CalcAverageByKey<ZipIterator, Eigen::Vector3f_u>(keys,
                    thrust::make_zip_iterator(thrust::make_tuple(sorted_points.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(output->points_.begin())));
        output->points_.resize(n_out);
    } else if (has_normals && !has_colors) {
        thrust::device_vector<Eigen::Vector3f_u> sorted_normals = normals_;
        output->normals_.resize(n);
        typedef thrust::tuple<thrust::device_vector<Eigen::Vector3f_u>::iterator, thrust::device_vector<Eigen::Vector3f_u>::iterator> IteratorTuple;
        typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
        auto n_out = CalcAverageByKey<ZipIterator, Eigen::Vector3f_u, Eigen::Vector3f_u>(keys,
                    thrust::make_zip_iterator(thrust::make_tuple(sorted_points.begin(), sorted_normals.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(output->points_.begin(), output->normals_.begin())));
        output->points_.resize(n_out);
        output->normals_.resize(n_out);
        thrust::for_each(output->normals_.begin(), output->normals_.end(), [] __device__ (Eigen::Vector3f_u& nl) {nl.normalize();});
    } else if (!has_normals && has_colors) {
        thrust::device_vector<Eigen::Vector3f_u> sorted_colors = colors_;
        output->colors_.resize(n);
        typedef thrust::tuple<thrust::device_vector<Eigen::Vector3f_u>::iterator, thrust::device_vector<Eigen::Vector3f_u>::iterator> IteratorTuple;
        typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
        auto n_out = CalcAverageByKey<ZipIterator, Eigen::Vector3f_u, Eigen::Vector3f_u>(keys,
                    thrust::make_zip_iterator(thrust::make_tuple(sorted_points.begin(), sorted_colors.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(output->points_.begin(), output->colors_.begin())));
        output->points_.resize(n_out);
        output->colors_.resize(n_out);
    } else {
        thrust::device_vector<Eigen::Vector3f_u> sorted_normals = normals_;
        thrust::device_vector<Eigen::Vector3f_u> sorted_colors = colors_;
        output->normals_.resize(n);
        output->colors_.resize(n);
        typedef thrust::tuple<thrust::device_vector<Eigen::Vector3f_u>::iterator, thrust::device_vector<Eigen::Vector3f_u>::iterator, thrust::device_vector<Eigen::Vector3f_u>::iterator> IteratorTuple;
        typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
        auto n_out = CalcAverageByKey<ZipIterator, Eigen::Vector3f_u, Eigen::Vector3f_u, Eigen::Vector3f_u>(keys,
                    thrust::make_zip_iterator(thrust::make_tuple(sorted_points.begin(), sorted_normals.begin(), sorted_colors.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(output->points_.begin(), output->normals_.begin(), output->colors_.begin())));
        output->points_.resize(n_out);
        output->normals_.resize(n_out);
        output->colors_.resize(n_out);
        thrust::for_each(output->normals_.begin(), output->normals_.end(), [] __device__ (Eigen::Vector3f_u& nl) {nl.normalize();});
    }

    utility::LogDebug(
            "Pointcloud down sampled from {:d} points to {:d} points.\n",
            (int)points_.size(), (int)output->points_.size());
    return output;
}
