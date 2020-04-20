#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/geometry/intersection_test.h"
#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/geometry_functor.h"

#include "cupoch/utility/eigen.h"

namespace cupoch {
namespace geometry {

namespace {

struct compute_free_voxels_functor{
    compute_free_voxels_functor(const Eigen::Vector3f* points,
                                const Eigen::Vector3f& viewpoint,
                                const Eigen::Vector3f& min_bound,
                                float voxel_size,
                                const Eigen::Vector3f& origin,
                                int num_h, int num_d, int n_points)
                                : points_(points), viewpoint_(viewpoint),
                                 min_bound_(min_bound), voxel_size_(voxel_size),
                                 box_half_size_(Eigen::Vector3f(
                                    voxel_size / 2, voxel_size / 2, voxel_size / 2)),
                                 origin_(origin),
                                 num_h_(num_h), num_d_(num_d), n_points_(n_points) {};
    const Eigen::Vector3f* points_;
    const Eigen::Vector3f viewpoint_;
    const Eigen::Vector3f min_bound_;
    const float voxel_size_;
    const Eigen::Vector3f box_half_size_;
    const Eigen::Vector3f origin_;
    const int num_h_;
    const int num_d_;
    const int n_points_;
    __device__ Eigen::Vector3i operator() (size_t idx) {
        int widx = idx / (num_h_ * num_d_ * n_points_);
        int hdpidx = idx % (num_h_ * num_d_ * n_points_);
        int hidx = hdpidx / (num_d_ * n_points_);
        int dpidx = hdpidx % (num_d_ * n_points_);
        int didx = dpidx / n_points_;
        int pidx = dpidx % n_points_;
        Eigen::Vector3f center = Eigen::Vector3f(widx, hidx, didx) * voxel_size_ + min_bound_ - box_half_size_;
        if (intersection_test::LineSegmentAABB(viewpoint_, points_[pidx],
                                               center - box_half_size_,
                                               center + box_half_size_)) {
            return Eigen::device_floor<Eigen::Vector3f>((center - origin_) / voxel_size_).cast<int>();
        } else {
            return Eigen::Vector3i(geometry::INVALID_VOXEL_INDEX,
                                   geometry::INVALID_VOXEL_INDEX,
                                   geometry::INVALID_VOXEL_INDEX);
        }
    }
};

void ComputeFreeVoxels(const utility::device_vector<Eigen::Vector3f>& points,
                       const Eigen::Vector3f& viewpoint,
                       float voxel_size, Eigen::Vector3f& origin,
                       utility::device_vector<Eigen::Vector3i>& free_voxels) {
    size_t n_points = points.size();
    auto bbx = AxisAlignedBoundingBox::CreateFromPoints(points);
    Eigen::Vector3f min_bound = viewpoint.array().min(bbx.min_bound_.array());
    Eigen::Vector3f max_bound = viewpoint.array().max(bbx.max_bound_.array());
    Eigen::Vector3f grid_size = max_bound - min_bound;
    int num_w = int(std::round(grid_size(0) / voxel_size));
    int num_h = int(std::round(grid_size(1) / voxel_size));
    int num_d = int(std::round(grid_size(2) / voxel_size));
    size_t n_total = num_w * num_h * num_d * n_points;

    free_voxels.resize(n_total);
    compute_free_voxels_functor func(thrust::raw_pointer_cast(points.data()),
                                     viewpoint, min_bound,
                                     voxel_size, origin,
                                     num_h, num_d, n_points);
    thrust::transform(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(n_total),
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
    __device__ Eigen::Vector3i operator()(const Eigen::Vector3f &point) const {
        Eigen::Vector3f ref_coord = (point - origin_) / voxel_size_;
        return Eigen::device_floor<Eigen::Vector3f>(ref_coord).cast<int>();
    }
};

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
OccupancyGrid::~OccupancyGrid() {}
OccupancyGrid::OccupancyGrid(const OccupancyGrid& other)
 : Geometry3D(Geometry::GeometryType::OccupancyGrid), voxel_size_(other.voxel_size_),
   origin_(other.origin_), voxels_keys_(other.voxels_keys_), voxels_values_(other.voxels_values_),
   clamping_thres_min_(other.clamping_thres_min_), clamping_thres_max_(other.clamping_thres_max_),
   prob_hit_log_(other.prob_hit_log_), prob_miss_log_(other.prob_miss_log_),
   occ_prob_thres_log_(other.occ_prob_thres_log_) {}

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

void OccupancyGrid::Insert(const utility::device_vector<Eigen::Vector3f>& points, const Eigen::Vector3f& viewpoint) {
    utility::device_vector<Eigen::Vector3i> free_voxels;
    utility::device_vector<Eigen::Vector3i> occupied_voxels;

    // comupute free voxels
    ComputeFreeVoxels(points, viewpoint, voxel_size_, origin_, free_voxels); 

    { // compute occupied voxels
        occupied_voxels.resize(points.size());
        create_occupancy_voxels_functor func(origin_, voxel_size_);
        thrust::transform(points.begin(), points.end(), occupied_voxels.begin(), func);
        thrust::sort(occupied_voxels.begin(), occupied_voxels.end());
        auto end2 = thrust::unique(occupied_voxels.begin(), occupied_voxels.end());
        occupied_voxels.resize(thrust::distance(occupied_voxels.begin(), end2));
    }
    utility::device_vector<Eigen::Vector3i> free_voxels_res(free_voxels.size());
    auto end = thrust::set_difference(free_voxels.begin(), free_voxels.end(),
                                      occupied_voxels.begin(), occupied_voxels.end(),
                                      free_voxels_res.begin());
    free_voxels_res.resize(thrust::distance(free_voxels_res.begin(), end));
    AddVoxels(free_voxels_res, false);
    AddVoxels(occupied_voxels, true);
}

void OccupancyGrid::Insert(const geometry::PointCloud& pointcloud, const Eigen::Vector3f& viewpoint) {
    Insert(pointcloud.points_, viewpoint);
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