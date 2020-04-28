#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/geometry/intersection_test.h"
#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/geometry_functor.h"

#include "cupoch/utility/eigen.h"

namespace cupoch {
namespace geometry {

namespace {

__constant__ float voxel_offset[7][3] = {{0, 0, 0}, {1, 0, 0}, {-1, 0, 0},
                                         {0, 1, 0}, {0, -1, 0}, {0, 0, 1},
                                         {0, 0, -1}};

struct compute_intersect_voxel_segment_functor{
    compute_intersect_voxel_segment_functor(const Eigen::Vector3f* points,
                                            const Eigen::Vector3f* steps,
                                            const Eigen::Vector3f& viewpoint,
                                            const Eigen::Vector3f& min_bound,
                                            float voxel_size,
                                            const Eigen::Vector3f& origin,
                                            int n_div)
                                            : points_(points), steps_(steps), viewpoint_(viewpoint),
                                             min_bound_(min_bound), voxel_size_(voxel_size),
                                             box_half_size_(Eigen::Vector3f(
                                                voxel_size / 2, voxel_size / 2, voxel_size / 2)),
                                             origin_(origin), n_div_(n_div) {};
    const Eigen::Vector3f* points_;
    const Eigen::Vector3f* steps_;
    const Eigen::Vector3f viewpoint_;
    const Eigen::Vector3f min_bound_;
    const float voxel_size_;
    const Eigen::Vector3f box_half_size_;
    const Eigen::Vector3f origin_;
    const int n_div_;
    __device__ Eigen::Vector3i operator() (size_t idx) {
        int pidx = idx / (n_div_ * 7);
        int svidx = idx % (n_div_ * 7);
        int sidx = svidx / 7;
        int vidx = svidx % 7;
        Eigen::Vector3f center = sidx * steps_[pidx] + viewpoint_;
        Eigen::Vector3f voxel_idx = Eigen::device_vectorize<float, 3, ::floor>((center - origin_) / voxel_size_);
        Eigen::Vector3f voxel_center = voxel_size_ * (voxel_idx + Eigen::Vector3f(voxel_offset[vidx][0], voxel_offset[vidx][1], voxel_offset[vidx][2]));
        bool is_intersect = intersection_test::LineSegmentAABB(viewpoint_, points_[pidx],
                                                               voxel_center - box_half_size_,
                                                               voxel_center + box_half_size_);
        return (is_intersect) ? voxel_idx.cast<int>() :
            Eigen::Vector3i(geometry::INVALID_VOXEL_INDEX, geometry::INVALID_VOXEL_INDEX, geometry::INVALID_VOXEL_INDEX);
    }
};

void ComputeFreeVoxels(const utility::device_vector<Eigen::Vector3f>& points,
                       const Eigen::Vector3f& viewpoint,
                       float voxel_size, Eigen::Vector3f& origin,
                       const utility::device_vector<Eigen::Vector3f>& steps, int n_div,
                       utility::device_vector<Eigen::Vector3i>& free_voxels) {
    if (points.empty()) return;
    size_t n_points = points.size();
    auto bbx = AxisAlignedBoundingBox::CreateFromPoints(points);
    const Eigen::Vector3f box_half_size(0.5 * voxel_size, 0.5 * voxel_size, 0.5 * voxel_size);
    Eigen::Vector3f min_bound = viewpoint.array().min(bbx.min_bound_.array()).matrix() - box_half_size;
    free_voxels.resize(n_div * n_points * 7);
    compute_intersect_voxel_segment_functor func(thrust::raw_pointer_cast(points.data()),
                                                 thrust::raw_pointer_cast(steps.data()),
                                                 viewpoint, min_bound,
                                                 voxel_size, origin, n_div);
    thrust::transform(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(n_div * n_points * 7),
                      free_voxels.begin(), func);
    auto end1 = thrust::remove_if(free_voxels.begin(), free_voxels.end(),
             [] __device__(
                    const Eigen::Vector3i &idx)
                    -> bool {
                return idx == Eigen::Vector3i(INVALID_VOXEL_INDEX,
                                              INVALID_VOXEL_INDEX,
                                              INVALID_VOXEL_INDEX);
            });
    free_voxels.resize(thrust::distance(free_voxels.begin(), end1));
    thrust::sort(free_voxels.begin(), free_voxels.end());
    auto end2 = thrust::unique(free_voxels.begin(), free_voxels.end());
    free_voxels.resize(thrust::distance(free_voxels.begin(), end2));
}

struct create_occupancy_voxels_functor {
    create_occupancy_voxels_functor(const Eigen::Vector3f &origin,
                                    float voxel_size)
        : origin_(origin),
          voxel_size_(voxel_size) {};
    const Eigen::Vector3f origin_;
    const float voxel_size_;
    __device__ Eigen::Vector3i operator()(const thrust::tuple<Eigen::Vector3f, bool> &x) const {
        const Eigen::Vector3f& point = thrust::get<0>(x);
        bool hit_flag = thrust::get<1>(x);
        Eigen::Vector3f ref_coord = (point - origin_) / voxel_size_;
        return (hit_flag) ? Eigen::device_vectorize<float, 3, ::floor>(ref_coord).cast<int>() :
            Eigen::Vector3i(INVALID_VOXEL_INDEX,
                INVALID_VOXEL_INDEX,
                INVALID_VOXEL_INDEX);;
    }
};

void ComputeOccupiedVoxels(const utility::device_vector<Eigen::Vector3f>& points,
    const utility::device_vector<bool> hit_flags,
    float voxel_size, Eigen::Vector3f& origin,
    utility::device_vector<Eigen::Vector3i>& occupied_voxels) {
    occupied_voxels.resize(points.size());
    create_occupancy_voxels_functor func(origin, voxel_size);
    thrust::transform(make_tuple_iterator(points.begin(), hit_flags.begin()),
                      make_tuple_iterator(points.end(), hit_flags.end()),
                      occupied_voxels.begin(), func);
    auto end1 = thrust::remove_if(occupied_voxels.begin(), occupied_voxels.end(),
             [] __device__(
                    const Eigen::Vector3i &idx)
                    -> bool {
                return idx == Eigen::Vector3i(INVALID_VOXEL_INDEX,
                                              INVALID_VOXEL_INDEX,
                                              INVALID_VOXEL_INDEX);
            });
    occupied_voxels.resize(thrust::distance(occupied_voxels.begin(), end1));
    thrust::sort(occupied_voxels.begin(), occupied_voxels.end());
    auto end2 = thrust::unique(occupied_voxels.begin(), occupied_voxels.end());
    occupied_voxels.resize(thrust::distance(occupied_voxels.begin(), end2));
}

struct create_occupancy_voxel_functor{
    create_occupancy_voxel_functor(float prob_hit_log,
                                   float prob_miss_log,
                                   bool occupied)
                                   : prob_hit_log_(prob_hit_log),
                                   prob_miss_log_(prob_miss_log),
                                   occupied_(occupied) {};
    const float prob_hit_log_;
    const float prob_miss_log_;
    const bool occupied_;
    __device__ OccupancyVoxel operator() (const Eigen::Vector3i& idx) const {
        return OccupancyVoxel(idx, (occupied_)? prob_hit_log_ : prob_miss_log_);
    }
};

struct add_occupancy_functor{
    add_occupancy_functor(float clamping_thres_min, float clamping_thres_max)
     : clamping_thres_min_(clamping_thres_min), clamping_thres_max_(clamping_thres_max) {};
    const float clamping_thres_min_;
    const float clamping_thres_max_;
    __device__ OccupancyVoxel operator() (const OccupancyVoxel& lhs, const OccupancyVoxel& rhs) const {
        float sum_prob = lhs.prob_log_ + rhs.prob_log_;
        return OccupancyVoxel(lhs.grid_index_,
                              min(max(sum_prob, clamping_thres_min_), clamping_thres_max_),
                              (lhs.color_ + rhs.color_) * 0.5);
    }
};

}

OccupancyGrid::OccupancyGrid() : Geometry3D(Geometry::GeometryType::OccupancyGrid) {}
OccupancyGrid::OccupancyGrid(float voxel_size, const Eigen::Vector3f& origin)
 : Geometry3D(Geometry::GeometryType::OccupancyGrid), voxel_size_(voxel_size), origin_(origin) {}
OccupancyGrid::~OccupancyGrid() {}
OccupancyGrid::OccupancyGrid(const OccupancyGrid& other)
 : Geometry3D(Geometry::GeometryType::OccupancyGrid), voxel_size_(other.voxel_size_),
   origin_(other.origin_), voxels_keys_(other.voxels_keys_), voxels_values_(other.voxels_values_),
   clamping_thres_min_(other.clamping_thres_min_), clamping_thres_max_(other.clamping_thres_max_),
   prob_hit_log_(other.prob_hit_log_), prob_miss_log_(other.prob_miss_log_),
   occ_prob_thres_log_(other.occ_prob_thres_log_), visualize_free_area_(other.visualize_free_area_) {}

OccupancyGrid &OccupancyGrid::Clear() {
    voxel_size_ = 0.0;
    origin_ = Eigen::Vector3f::Zero();
    voxels_keys_.clear();
    voxels_values_.clear();
    return *this;
}

bool OccupancyGrid::IsEmpty() const { return !HasVoxels(); }

Eigen::Vector3f OccupancyGrid::GetMinBound() const {
    if (!HasVoxels()) {
        return origin_;
    } else {
        Eigen::Vector3i init = voxels_keys_[0];
        Eigen::Vector3i min_grid_index = thrust::reduce(voxels_keys_.begin(),
                voxels_keys_.end(), init, thrust::elementwise_minimum<Eigen::Vector3i>());
        return min_grid_index.cast<float>() * voxel_size_ + origin_;
    }
}

Eigen::Vector3f OccupancyGrid::GetMaxBound() const {
    if (!HasVoxels()) {
        return origin_;
    } else {
        Eigen::Vector3i init = voxels_keys_[0];
        Eigen::Vector3i max_grid_index = thrust::reduce(voxels_keys_.begin(),
                voxels_keys_.end(), init, thrust::elementwise_maximum<Eigen::Vector3i>());
        return (max_grid_index.cast<float>() + Eigen::Vector3f::Ones()) *
                       voxel_size_ +
               origin_;
    }
}

Eigen::Vector3f OccupancyGrid::GetCenter() const {
    Eigen::Vector3f init(0, 0, 0);
    if (!HasVoxels()) {
        return init;
    }
    compute_grid_center_functor func(voxel_size_, origin_);
    Eigen::Vector3f center = thrust::transform_reduce(voxels_keys_.begin(),
            voxels_keys_.end(), func, init, thrust::plus<Eigen::Vector3f>());
    center /= float(voxels_values_.size());
    return center;
}

AxisAlignedBoundingBox OccupancyGrid::GetAxisAlignedBoundingBox() const {
    AxisAlignedBoundingBox box;
    box.min_bound_ = GetMinBound();
    box.max_bound_ = GetMaxBound();
    return box;
}

OrientedBoundingBox OccupancyGrid::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(
            GetAxisAlignedBoundingBox());
}

bool OccupancyGrid::IsOccupied(const Eigen::Vector3f &point) const{
    auto idx = GetVoxelIndex(point);
    if (idx < 0) return false;
    OccupancyVoxel voxel = voxels_values_[idx];
    return voxel.prob_log_ > occ_prob_thres_log_;
}

bool OccupancyGrid::IsUnknown(const Eigen::Vector3f &point) const{
    auto idx = GetVoxelIndex(point);
    return idx < 0;
}

int OccupancyGrid::GetVoxelIndex(const Eigen::Vector3f& point) const {
    Eigen::Vector3f voxel_f = (point - origin_) / voxel_size_;
    Eigen::Vector3i voxel_idx = (Eigen::floor(voxel_f.array())).cast<int>();
    auto itr = thrust::find(voxels_keys_.begin(), voxels_keys_.end(), voxel_idx);
    if (itr == voxels_keys_.end()) return -1;
    return thrust::distance(voxels_keys_.begin(), itr);
}

thrust::tuple<bool, OccupancyVoxel> OccupancyGrid::GetVoxel(const Eigen::Vector3f &point) const {
    auto idx = GetVoxelIndex(point);
    if (idx < 0) return thrust::make_tuple(false, OccupancyVoxel());
    OccupancyVoxel voxel = voxels_values_[idx];
    return thrust::make_tuple(true, voxel);
}

OccupancyGrid &OccupancyGrid::Transform(const Eigen::Matrix4f &transformation) {
    utility::LogError("OccupancyGrid::Transform is not supported");
    return *this;
}

OccupancyGrid &OccupancyGrid::Translate(const Eigen::Vector3f &translation,
                                        bool relative) {
    origin_ += translation;
    return *this;
}

OccupancyGrid &OccupancyGrid::Scale(const float scale, bool center) {
    voxel_size_ *= scale;
    return *this;
}

OccupancyGrid &OccupancyGrid::Rotate(const Eigen::Matrix3f &R, bool center) {
    utility::LogError("OccupancyGrid::Rotate is not supported");
    return *this;
}

OccupancyGrid& OccupancyGrid::Insert(const utility::device_vector<Eigen::Vector3f>& points,
                                     const Eigen::Vector3f& viewpoint, float max_range) {
    if (points.empty()) return *this;

    utility::device_vector<Eigen::Vector3f> ranged_points(points.size());
    utility::device_vector<float> ranged_dists(points.size());
    utility::device_vector<bool> hit_flags(points.size());

    thrust::transform(points.begin(), points.end(),
                      make_tuple_iterator(ranged_points.begin(), ranged_dists.begin(), hit_flags.begin()),
                      [viewpoint, max_range] __device__ (const Eigen::Vector3f &pt) {
                          Eigen::Vector3f pt_vp = pt - viewpoint;
                          float dist = pt_vp.norm();
                          bool is_hit = max_range < 0 || dist <= max_range;
                          return thrust::make_tuple((is_hit) ? pt : viewpoint + pt_vp / dist * max_range,
                                                    (is_hit) ? dist : max_range, is_hit);
                      });
    float max_dist = *(thrust::max_element(ranged_dists.begin(), ranged_dists.end()));
    int n_div = int(std::ceil(max_dist / voxel_size_));

    utility::device_vector<Eigen::Vector3i> free_voxels;
    utility::device_vector<Eigen::Vector3i> occupied_voxels;
    if (n_div > 0) {
        utility::device_vector<Eigen::Vector3f> steps(points.size());
        thrust::transform(ranged_points.begin(), ranged_points.end(), steps.begin(),
                          [viewpoint, n_div] __device__ (const Eigen::Vector3f& pt) {
                              return (pt - viewpoint) / n_div;
                          });
        // comupute free voxels
        ComputeFreeVoxels(ranged_points, viewpoint, voxel_size_, origin_, steps, n_div + 1, free_voxels);
    } else {
        thrust::copy(points.begin(), points.end(), ranged_points.begin());
        thrust::fill(hit_flags.begin(), hit_flags.end(), true);
    }
    // compute occupied voxels
    ComputeOccupiedVoxels(ranged_points, hit_flags, voxel_size_, origin_, occupied_voxels);

    if (n_div > 0) {
        utility::device_vector<Eigen::Vector3i> free_voxels_res(free_voxels.size());
        auto end = thrust::set_difference(free_voxels.begin(), free_voxels.end(),
                                          occupied_voxels.begin(), occupied_voxels.end(),
                                          free_voxels_res.begin());
        free_voxels_res.resize(thrust::distance(free_voxels_res.begin(), end));
        AddVoxels(free_voxels_res, false);
    }
    AddVoxels(occupied_voxels, true);
    return *this;
}

OccupancyGrid& OccupancyGrid::Insert(const thrust::host_vector<Eigen::Vector3f>& points,
                                     const Eigen::Vector3f& viewpoint, float max_range) {
    utility::device_vector<Eigen::Vector3f> dev_points = points;
    return Insert(dev_points, viewpoint, max_range);
}

OccupancyGrid& OccupancyGrid::Insert(const geometry::PointCloud& pointcloud,
                                     const Eigen::Vector3f& viewpoint, float max_range) {
    Insert(pointcloud.points_, viewpoint, max_range);
    return *this;
}

void OccupancyGrid::AddVoxel(const Eigen::Vector3i &voxel, bool occupied) {
    voxels_keys_.push_back(voxel);
    voxels_values_.push_back(OccupancyVoxel(voxel, (occupied) ? prob_hit_log_ : prob_miss_log_));
    thrust::sort_by_key(voxels_keys_.begin(), voxels_keys_.end(),
                        voxels_values_.begin());
    utility::device_vector<Eigen::Vector3i> new_voxels_keys(voxels_keys_.size());
    utility::device_vector<OccupancyVoxel> new_voxels_values(voxels_keys_.size());
    auto end = thrust::reduce_by_key(voxels_keys_.begin(), voxels_keys_.end(),
                                     voxels_values_.begin(), new_voxels_keys.begin(),
                                     new_voxels_values.begin(), thrust::equal_to<Eigen::Vector3i>(),
                                     add_occupancy_functor(clamping_thres_min_, clamping_thres_max_));
    size_t out_size = thrust::distance(new_voxels_keys.begin(), end.first);
    new_voxels_keys.resize(out_size);
    new_voxels_values.resize(out_size);
    voxels_keys_ = new_voxels_keys;
    voxels_values_ = new_voxels_values;
}

void OccupancyGrid::AddVoxels(const utility::device_vector<Eigen::Vector3i>& voxels, bool occupied) {
    voxels_keys_.insert(voxels_keys_.end(), voxels.begin(), voxels.end());
    create_occupancy_voxel_functor func(prob_hit_log_, prob_miss_log_, occupied);
    voxels_values_.insert(voxels_values_.end(),
                          thrust::make_transform_iterator(voxels.begin(), func),
                          thrust::make_transform_iterator(voxels.end(), func));
    thrust::sort_by_key(voxels_keys_.begin(), voxels_keys_.end(),
                        voxels_values_.begin());
    utility::device_vector<Eigen::Vector3i> new_voxels_keys(voxels_keys_.size());
    utility::device_vector<OccupancyVoxel> new_voxels_values(voxels_keys_.size());
    auto end = thrust::reduce_by_key(voxels_keys_.begin(), voxels_keys_.end(),
                                     voxels_values_.begin(), new_voxels_keys.begin(),
                                     new_voxels_values.begin(), thrust::equal_to<Eigen::Vector3i>(),
                                     add_occupancy_functor(clamping_thres_min_, clamping_thres_max_));
    size_t out_size = thrust::distance(new_voxels_keys.begin(), end.first);
    new_voxels_keys.resize(out_size);
    new_voxels_values.resize(out_size);
    voxels_keys_ = new_voxels_keys;
    voxels_values_ = new_voxels_values;
}

}
}