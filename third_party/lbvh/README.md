# LBVH

An implementation of the following paper

- Tero Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees", High Performance Graphics (2012)

and the following blog posts

- https://devblogs.nvidia.com/thinking-parallel-part-ii-tree-traversal-gpu/
- https://devblogs.nvidia.com/thinking-parallel-part-iii-tree-construction-gpu/

depending on [thrust](https://thrust.github.io/).

It can contain any object and also handles morton code overlap.

If the morton codes of objects are the same, it appends object index to the
morton code and use the "extended" indices as described in the paper.

Also, nearest neighbor query based on the following paper is supported.

- Nick Roussopoulos, Stephen Kelley, Frederic Vincent, "Nearest Neighbor Queries", ACM-SIGMOD (1995)

## example code

```cpp
struct object
{
    // any object you want.
    // For example, sphere.
    float4 xyzr;
};

struct aabb_getter
{
    // return aabb<float> if your object uses float.
    // if you chose double, return aabb<double>.
    __device__
    lbvh::aabb<float> operator()(const object& f) const noexcept
    {
        // calculate aabb of object ...
        const float r = f.xyzr.r;
        lbvh::aabb<float> box;
        box.upper = make_float4(f.xyzr.x + r, f.xyzr.y + r, f.xyzr.z + r, 0.0f);
        box.lower = make_float4(f.xyzr.x - r, f.xyzr.y - r, f.xyzr.z - r, 0.0f);
        return box;
    }
};

// this struct will be used in nearest neighbor query (If you don't need nearest
// neighbor query, you don't need to implement this).
struct distance_sq_calculator
{
    __device__
    float operator()(const float4 pos, const object& f) const noexcept
    {
        // calculate square distance...
        const float dx = pos.x - f.xyzr.x;
        const float dy = pos.y - f.xyzr.y;
        const float dz = pos.z - f.xyzr.z;
        return dx * dx + dy * dy + dz * dz;
    }
};

int main()
{
    std::vector<objects> objs;

    // initialize objs ...

    // construct a bvh
    lbvh::bvh<float, object, aabb_getter> bvh(objs.begin(), objs.end());

    // get a set of device (raw) pointers to use it in device functions.
    // Do not use this on host!
    const auto bvh_dev = bvh.get_device_repr();

    thrust::for_each(thrust::device,
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(N),
        [bvh_dev] __device__ (const unsigned int idx)
        {
            unsigned int buffer[10];

            point_getter get_point;
            const auto self = get_point(bvh_dev.objects[idx]);

            // make a query box.
            const lbvh::aabb<float> box(
                    make_float4(self.x-0.1, self.y-0.1, self.z-0.1, 0),
                    make_float4(self.x+0.1, self.y+0.1, self.z+0.1, 0)
                );

            // send a query!
            const auto num_found = query_device(bvh_dev, overlaps(box), 10, buffer);

            for(unsigned int j=0; j<num_found; ++j)
            {
                const unsigned int other_idx = buffer[j];
                const object&      other     = bvh_dev.objects[other_idx];
                // do some stuff ...
            }

            const float3 pos = make_float3(0.0, 1.0, 2.0);
            const auto nearest = query_device(bvh_dev, nearest(pos), distance_sq_calculator());

            return ;
        });

    return 0;
}
```

## Synopsis

### AABB

```cpp
template<typename T>
struct aabb
{
    /* T4 (float4 if T == float, double4 if T == double) */ upper;
    /* T4 (float4 if T == float, double4 if T == double) */ lower;
};

template<typename T>
__device__ __host__
inline bool intersects(const aabb<T>& lhs, const aabb<T>& rhs) noexcept;


template<typename T>
__device__ __host__
inline aabb<T> merge(const aabb<T>& lhs, const aabb<T>& rhs) noexcept
```

### BVH

```cpp
template<typename Real, typename Object, typename AABBGetter,
         typename MortonCodeCalculator = default_morton_code_calculator<Real, Object, AABBGetter>>
class bvh
{
  public:
    using real_type   = Real;
    using index_type = std::uint32_t;
    using object_type = Object;
    using aabb_type   = aabb<real_type>;
    using node_type   = detail::node;
    using point_getter_type = PointGetter;
    using aabb_getter_type  = AABBGetter;

    template<typename InputIterator>
    bvh(InputIterator first, InputIterator last)
        : objects_h_(first, last), objects_d_(objects_h_)
    {
        this->construct();
    }

    template<typename InputIterator>
    void assign(InputIterator first, InputIterator last);
    void clear();

    bvh_device<real_type, object_type>  get_device_repr()       noexcept;
    cbvh_device<real_type, object_type> get_device_repr() const noexcept;
};
namespace detail {
template<typename Real, typename Object, bool IsConst>
struct basic_device_bvh
{
    using real_type  = Real;
    using aabb_type  = aabb<real_type>;
    using node_type  = detail::node;
    using index_type = std::uint32_t;
    using object_type = Object;

    unsigned int num_nodes;   // (# of internal node) + (# of leaves), 2N+1
    unsigned int num_leaves;  // (# of leaves), N
    unsigned int num_objects; // (# of objects) ; can be larger than N

    /* const if IsConst is true */ node_type  * nodes;
    /* const if IsConst is true */ aabb_type  * aabbs;
    /* const if IsConst is true */ index_type * ranges;
    /* const if IsConst is true */ index_type * indices;
    /* const if IsConst is true */ object_type* objects;
};
}
template<typename Real, typename Object>
using  bvh_device = detail::basic_device_bvh<Real, Object, false>;
template<typename Real, typename Object>
using cbvh_device = detail::basic_device_bvh<Real, Object, true>;
```

## queries

```cpp
template<typename Real>
__device__ __host__
query_overlap<Real> overlaps(const aabb<Real>& region) noexcept;

template<typename Real>
__device__ __host__
inline query_nearest<Real> nearest(const /*Real4 or Real3*/& point) noexcept
{
    return query_nearest<Real>(point);
}

template<typename Real, typename Objects, bool IsConst, typename OutputIterator>
__device__
unsigned int query_device(
        const detail::basic_device_bvh<Real, Objects, IsConst>& bvh,
        const query_overlap<Real>& q, unsigned int max_buffer_size,
        OutputIterator outiter) noexcept;

template<typename Real, typename Objects, bool IsConst, typename DistanceCalculator>
__device__
thrust::pair<unsigned int, Real> query_device(
        const detail::basic_device_bvh<Real, Objects, IsConst>& bvh,
        const query_nearest<Real>& q, DistanceCalculator calc_dist) noexcept
```
