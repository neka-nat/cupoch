#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/utility/console.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct compute_sphere_vertices_functor {
    compute_sphere_vertices_functor(int resolution, float radius)
        : resolution_(resolution), radius_(radius) {
        step_ = M_PI / (float)resolution;
    };
    const int resolution_;
    const float radius_;
    float step_;
    __device__ Eigen::Vector3f operator()(size_t idx) const {
        int i = idx / (2 * resolution_) + 1;
        int j = idx % (2 * resolution_);

        float alpha = step_ * i;
        float theta = step_ * j;
        return Eigen::Vector3f(sin(alpha) * cos(theta), sin(alpha) * sin(theta),
                               cos(alpha)) *
               radius_;
    }
};

struct compute_sphere_triangles_functor1 {
    compute_sphere_triangles_functor1(Eigen::Vector3i *triangle, int resolution)
        : triangles_(triangle), resolution_(resolution){};
    Eigen::Vector3i *triangles_;
    const int resolution_;
    __device__ void operator()(size_t idx) {
        int j1 = (idx + 1) % (2 * resolution_);
        int base = 2;
        triangles_[2 * idx] = Eigen::Vector3i(0, base + idx, base + j1);
        base = 2 + 2 * resolution_ * (resolution_ - 2);
        triangles_[2 * idx + 1] = Eigen::Vector3i(1, base + j1, base + idx);
    }
};

struct compute_sphere_triangles_functor2 {
    compute_sphere_triangles_functor2(Eigen::Vector3i *triangle, int resolution,
                                      int initial_base = 2)
        : triangles_(triangle), resolution_(resolution),
        initial_base_(initial_base) {};
    Eigen::Vector3i *triangles_;
    const int resolution_;
    const int initial_base_;
    __device__ void operator()(size_t idx) {
        int i = idx / (2 * resolution_) + 1;
        int j = idx % (2 * resolution_);
        int base1 = initial_base_ + 2 * resolution_ * (i - 1);
        int base2 = base1 + 2 * resolution_;
        int j1 = (j + 1) % (2 * resolution_);
        triangles_[2 * idx] = Eigen::Vector3i(base2 + j, base1 + j1, base1 + j);
        triangles_[2 * idx + 1] =
                Eigen::Vector3i(base2 + j, base2 + j1, base1 + j1);
    }
};

struct compute_half_sphere_triangles_functor1 {
    compute_half_sphere_triangles_functor1(Eigen::Vector3i *triangle, int resolution)
        : triangles_(triangle), resolution_(resolution){};
    Eigen::Vector3i *triangles_;
    const int resolution_;
    __device__ void operator()(size_t idx) {
        int j1 = (idx + 1) % (2 * resolution_);
        int base = 1;
        triangles_[idx] = Eigen::Vector3i(0, base + idx, base + j1);
    }
};

struct compute_cylinder_vertices_functor {
    compute_cylinder_vertices_functor(int resolution,
                                      float radius,
                                      float height,
                                      float step,
                                      float h_step)
        : resolution_(resolution),
          radius_(radius),
          height_(height),
          step_(step),
          h_step_(h_step){};
    const int resolution_;
    const float radius_;
    const float height_;
    const float step_;
    const float h_step_;
    __device__ Eigen::Vector3f operator()(size_t idx) const {
        int i = idx / resolution_;
        int j = idx % resolution_;
        float theta = step_ * j;
        return Eigen::Vector3f(cos(theta) * radius_, sin(theta) * radius_,
                               height_ * 0.5 - h_step_ * i);
    }
};

struct compute_cylinder_triangles_functor1 {
    compute_cylinder_triangles_functor1(Eigen::Vector3i *triangle,
                                        int resolution,
                                        int split)
        : triangles_(triangle), resolution_(resolution), split_(split){};
    Eigen::Vector3i *triangles_;
    const int resolution_;
    const int split_;
    __device__ void operator()(size_t idx) {
        int j1 = (idx + 1) % resolution_;
        int base = 2;
        triangles_[2 * idx] = Eigen::Vector3i(0, base + idx, base + j1);
        base = 2 + resolution_ * split_;
        triangles_[2 * idx + 1] = Eigen::Vector3i(1, base + j1, base + idx);
    }
};

struct compute_cylinder_triangles_functor2 {
    compute_cylinder_triangles_functor2(Eigen::Vector3i *triangle,
                                        int resolution,
                                        int initial_base = 2)
        : triangles_(triangle), resolution_(resolution),
        initial_base_(initial_base) {};
    Eigen::Vector3i *triangles_;
    const int resolution_;
    const int initial_base_;
    __device__ void operator()(size_t idx) {
        int i = idx / resolution_;
        int j = idx % resolution_;
        int base1 = initial_base_ + resolution_ * i;
        int base2 = base1 + resolution_;
        int j1 = (j + 1) % resolution_;
        triangles_[2 * idx] = Eigen::Vector3i(base2 + j, base1 + j1, base1 + j);
        triangles_[2 * idx + 1] =
                Eigen::Vector3i(base2 + j, base2 + j1, base1 + j1);
    }
};

struct compute_cone_vertices_functor {
    compute_cone_vertices_functor(
            int resolution, int split, float step, float r_step, float h_step)
        : resolution_(resolution),
          split_(split),
          step_(step),
          r_step_(r_step),
          h_step_(h_step){};
    const int resolution_;
    const int split_;
    const float step_;
    const float r_step_;
    const float h_step_;
    __device__ Eigen::Vector3f operator()(size_t idx) const {
        int i = idx / resolution_;
        int j = idx % resolution_;
        float r = r_step_ * (split_ - i);
        float theta = step_ * j;
        return Eigen::Vector3f(cos(theta) * r, sin(theta) * r, h_step_ * i);
    }
};

struct compute_cone_triangles_functor1 {
    compute_cone_triangles_functor1(Eigen::Vector3i *triangle,
                                    int resolution,
                                    int split)
        : triangles_(triangle), resolution_(resolution), split_(split){};
    Eigen::Vector3i *triangles_;
    const int resolution_;
    const int split_;
    __device__ void operator()(size_t idx) {
        int j1 = (idx + 1) % resolution_;
        int base = 2;
        triangles_[2 * idx] = Eigen::Vector3i(0, base + j1, base + idx);
        base = 2 + resolution_ * (split_ - 1);
        triangles_[2 * idx + 1] = Eigen::Vector3i(1, base + idx, base + j1);
    }
};

struct compute_cone_triangles_functor2 {
    compute_cone_triangles_functor2(Eigen::Vector3i *triangle, int resolution)
        : triangles_(triangle), resolution_(resolution){};
    Eigen::Vector3i *triangles_;
    const int resolution_;
    __device__ void operator()(size_t idx) {
        int i = idx / resolution_;
        int j = idx % resolution_;
        int base1 = 2 + resolution_ * i;
        int base2 = base1 + resolution_;
        int j1 = (j + 1) % resolution_;
        triangles_[2 * idx] =
                Eigen::Vector3i(base2 + j1, base1 + j, base1 + j1);
        triangles_[2 * idx + 1] =
                Eigen::Vector3i(base2 + j1, base2 + j, base1 + j);
    }
};

struct compute_torus_mesh_functor {
    compute_torus_mesh_functor(Eigen::Vector3f* vertices, Eigen::Vector3i* triangles,
                               float torus_radius, float tube_radius,
                               int radial_resolution, int tubular_resolution)
                               : vertices_(vertices), triangles_(triangles), torus_radius_(torus_radius),
                               tube_radius_(tube_radius), radial_resolution_(radial_resolution),
                               tubular_resolution_(tubular_resolution),
                               u_step_(2 * M_PI / float(radial_resolution_)),
                               v_step_(2 * M_PI / float(tubular_resolution_)) {};
    Eigen::Vector3f* vertices_;
    Eigen::Vector3i* triangles_;
    const float torus_radius_;
    const float tube_radius_;
    const int radial_resolution_;
    const int tubular_resolution_;
    const float u_step_;
    const float v_step_;
    __device__
    int vert_idx(int uidx, int vidx) const {
        return uidx * tubular_resolution_ + vidx;
    };

    __device__
    void operator() (size_t idx) {
        int uidx = idx / tubular_resolution_;
        int vidx = idx % tubular_resolution_;
        float u = uidx * u_step_;
        Eigen::Vector3f w(cos(u), sin(u), 0);
        float v = vidx * v_step_;
        vertices_[vert_idx(uidx, vidx)] =
                torus_radius_ * w + tube_radius_ * cos(v) * w +
                Eigen::Vector3f(0, 0, tube_radius_ * sin(v));

        int tri_idx = (uidx * tubular_resolution_ + vidx) * 2;
        triangles_[tri_idx + 0] = Eigen::Vector3i(
                vert_idx((uidx + 1) % radial_resolution_, vidx),
                vert_idx((uidx + 1) % radial_resolution_,
                         (vidx + 1) % tubular_resolution_),
                vert_idx(uidx, vidx));
        triangles_[tri_idx + 1] = Eigen::Vector3i(
                vert_idx(uidx, vidx),
                vert_idx((uidx + 1) % radial_resolution_,
                         (vidx + 1) % tubular_resolution_),
                vert_idx(uidx, (vidx + 1) % tubular_resolution_));
    }
};

struct compute_moebius_vertices_functor {
    compute_moebius_vertices_functor(int length_split, int width_split,
                                     int twists, float radius,
                                     float flatness, float width,
                                     float scale)
                                     : width_split_(width_split),
                                     twists_(twists), radius_(radius), flatness_(flatness),
                                     width_(width), scale_(scale),
                                     u_step_(2 * M_PI / length_split),
                                     v_step_(width / (width_split - 1)) {};
    const int width_split_;
    const int twists_;
    const float radius_;
    const float flatness_;
    const float width_;
    const float scale_;
    const float u_step_;
    const float v_step_;
    __device__
    Eigen::Vector3f operator() (size_t idx) {
        int uidx = idx / width_split_;
        int vidx = idx % width_split_;
        float u = uidx * u_step_;
        float cos_u = cos(u);
        float sin_u = sin(u);
        float v = -width_ / 2.0 + vidx * v_step_;
        float alpha = twists_ * 0.5 * u;
        float cos_alpha = cos(alpha);
        float sin_alpha = sin(alpha);
        return Eigen::Vector3f(scale_ * ((cos_alpha * cos_u * v) + radius_ * cos_u),
                               scale_ * ((cos_alpha * sin_u * v) + radius_ * sin_u),
                               scale_ * sin_alpha * v * flatness_);
    }
};

struct compute_moebius_triangles_functor {
    compute_moebius_triangles_functor(Eigen::Vector3i* triangles,
                                      int length_split, int width_split,
                                      int twists)
        : triangles_(triangles), length_split_(length_split),
        width_split_(width_split), twists_(twists) {};
    Eigen::Vector3i* triangles_;
    const int length_split_;
    const int width_split_;
    const int twists_;
    __device__
    void operator() (size_t idx) {
        int uidx = idx / (width_split_ - 1);
        int vidx = idx % (width_split_ - 1);

        if (uidx == length_split_ - 1) {
            if (twists_ % 2 == 1) {
                if ((uidx + vidx) % 2 == 0) {
                    triangles_[idx * 2] =
                            Eigen::Vector3i((width_split_ - 1) - (vidx + 1),
                                            uidx * width_split_ + vidx,
                                            uidx * width_split_ + vidx + 1);
                    triangles_[idx * 2 + 1] = Eigen::Vector3i(
                            (width_split_ - 1) - vidx, uidx * width_split_ + vidx,
                            (width_split_ - 1) - (vidx + 1));
                } else {
                    triangles_[idx * 2] =
                            Eigen::Vector3i(uidx * width_split_ + vidx,
                                            uidx * width_split_ + vidx + 1,
                                            (width_split_ - 1) - vidx);
                    triangles_[idx * 2 + 1] = Eigen::Vector3i(
                            (width_split_ - 1) - vidx, uidx * width_split_ + vidx + 1,
                            (width_split_ - 1) - (vidx + 1));
                }
            } else {
                if ((uidx + vidx) % 2 == 0) {
                    triangles_[idx * 2] =
                            Eigen::Vector3i(uidx * width_split_ + vidx, vidx + 1,
                                            uidx * width_split_ + vidx + 1);
                    triangles_[idx * 2 + 1] = Eigen::Vector3i(
                            uidx * width_split_ + vidx, vidx, vidx + 1);
                } else {
                    triangles_[idx * 2] =
                            Eigen::Vector3i(uidx * width_split_ + vidx, vidx,
                                            uidx * width_split_ + vidx + 1);
                    triangles_[idx * 2 + 1] = Eigen::Vector3i(
                            uidx * width_split_ + vidx + 1, vidx, vidx + 1);
                }
            }
        } else {
            if ((uidx + vidx) % 2 == 0) {
                triangles_[idx * 2] =
                        Eigen::Vector3i(uidx * width_split_ + vidx,
                                        (uidx + 1) * width_split_ + vidx + 1,
                                        uidx * width_split_ + vidx + 1);
                triangles_[idx * 2 + 1] =
                        Eigen::Vector3i(uidx * width_split_ + vidx,
                                        (uidx + 1) * width_split_ + vidx,
                                        (uidx + 1) * width_split_ + vidx + 1);
            } else {
                triangles_[idx * 2] =
                        Eigen::Vector3i(uidx * width_split_ + vidx + 1,
                                        uidx * width_split_ + vidx,
                                        (uidx + 1) * width_split_ + vidx);
                triangles_[idx * 2 + 1] =
                        Eigen::Vector3i(uidx * width_split_ + vidx + 1,
                                        (uidx + 1) * width_split_ + vidx,
                                        (uidx + 1) * width_split_ + vidx + 1);
            }
        }
    }
};

}  // namespace

std::shared_ptr<TriangleMesh> TriangleMesh::CreateTetrahedron(
        float radius /* = 1.0*/) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateTetrahedron] radius <= 0");
    }
    mesh->vertices_.push_back(radius *
                              Eigen::Vector3f(std::sqrt(8. / 9.), 0, -1. / 3.));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(-std::sqrt(2. / 9.),
                                                       std::sqrt(2. / 3.),
                                                       -1. / 3.));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(-std::sqrt(2. / 9.),
                                                       -std::sqrt(2. / 3.),
                                                       -1. / 3.));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(0., 0., 1.));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 2, 1));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 3, 2));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 1, 3));
    mesh->triangles_.push_back(Eigen::Vector3i(1, 2, 3));
    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateOctahedron(
        float radius /* = 1.0*/) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateOctahedron] radius <= 0");
    }
    mesh->vertices_.push_back(radius * Eigen::Vector3f(1, 0, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(0, 1, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(0, 0, 1));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(-1, 0, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(0, -1, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(0, 0, -1));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 1, 2));
    mesh->triangles_.push_back(Eigen::Vector3i(1, 3, 2));
    mesh->triangles_.push_back(Eigen::Vector3i(3, 4, 2));
    mesh->triangles_.push_back(Eigen::Vector3i(4, 0, 2));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 5, 1));
    mesh->triangles_.push_back(Eigen::Vector3i(1, 5, 3));
    mesh->triangles_.push_back(Eigen::Vector3i(3, 5, 4));
    mesh->triangles_.push_back(Eigen::Vector3i(4, 5, 0));
    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateIcosahedron(
        float radius /* = 1.0*/) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateIcosahedron] radius <= 0");
    }
    const float p = (1. + std::sqrt(5.)) / 2.;
    mesh->vertices_.push_back(radius * Eigen::Vector3f(-1, 0, p));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(1, 0, p));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(1, 0, -p));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(-1, 0, -p));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(0, -p, 1));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(0, p, 1));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(0, p, -1));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(0, -p, -1));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(-p, -1, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(p, -1, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(p, 1, 0));
    mesh->vertices_.push_back(radius * Eigen::Vector3f(-p, 1, 0));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 4, 1));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 1, 5));
    mesh->triangles_.push_back(Eigen::Vector3i(1, 4, 9));
    mesh->triangles_.push_back(Eigen::Vector3i(1, 9, 10));
    mesh->triangles_.push_back(Eigen::Vector3i(1, 10, 5));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 8, 4));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 11, 8));
    mesh->triangles_.push_back(Eigen::Vector3i(0, 5, 11));
    mesh->triangles_.push_back(Eigen::Vector3i(5, 6, 11));
    mesh->triangles_.push_back(Eigen::Vector3i(5, 10, 6));
    mesh->triangles_.push_back(Eigen::Vector3i(4, 8, 7));
    mesh->triangles_.push_back(Eigen::Vector3i(4, 7, 9));
    mesh->triangles_.push_back(Eigen::Vector3i(3, 6, 2));
    mesh->triangles_.push_back(Eigen::Vector3i(3, 2, 7));
    mesh->triangles_.push_back(Eigen::Vector3i(2, 6, 10));
    mesh->triangles_.push_back(Eigen::Vector3i(2, 10, 9));
    mesh->triangles_.push_back(Eigen::Vector3i(2, 9, 7));
    mesh->triangles_.push_back(Eigen::Vector3i(3, 11, 6));
    mesh->triangles_.push_back(Eigen::Vector3i(3, 8, 11));
    mesh->triangles_.push_back(Eigen::Vector3i(3, 7, 8));
    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateBox(float width /* = 1.0*/,
                                                      float height /* = 1.0*/,
                                                      float depth /* = 1.0*/) {
    auto mesh_ptr = std::make_shared<TriangleMesh>();
    if (width <= 0) {
        utility::LogError("[CreateBox] width <= 0");
    }
    if (height <= 0) {
        utility::LogError("[CreateBox] height <= 0");
    }
    if (depth <= 0) {
        utility::LogError("[CreateBox] depth <= 0");
    }
    mesh_ptr->vertices_.resize(8);
    mesh_ptr->vertices_[0] = Eigen::Vector3f(0.0, 0.0, 0.0);
    mesh_ptr->vertices_[1] = Eigen::Vector3f(width, 0.0, 0.0);
    mesh_ptr->vertices_[2] = Eigen::Vector3f(0.0, 0.0, depth);
    mesh_ptr->vertices_[3] = Eigen::Vector3f(width, 0.0, depth);
    mesh_ptr->vertices_[4] = Eigen::Vector3f(0.0, height, 0.0);
    mesh_ptr->vertices_[5] = Eigen::Vector3f(width, height, 0.0);
    mesh_ptr->vertices_[6] = Eigen::Vector3f(0.0, height, depth);
    mesh_ptr->vertices_[7] = Eigen::Vector3f(width, height, depth);
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(4, 7, 5));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(4, 6, 7));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(0, 2, 4));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(2, 6, 4));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(0, 1, 2));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(1, 3, 2));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(1, 5, 7));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(1, 7, 3));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(2, 3, 7));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(2, 7, 6));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(0, 4, 1));
    mesh_ptr->triangles_.push_back(Eigen::Vector3i(1, 4, 5));
    return mesh_ptr;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateSphere(
        float radius /* = 1.0*/, int resolution /* = 20*/) {
    auto mesh_ptr = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateSphere] radius <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateSphere] resolution <= 0");
    }
    size_t n_vertices = 2 * resolution * (resolution - 1) + 2;
    mesh_ptr->vertices_.resize(n_vertices);
    mesh_ptr->vertices_[0] = Eigen::Vector3f(0.0, 0.0, radius);
    mesh_ptr->vertices_[1] = Eigen::Vector3f(0.0, 0.0, -radius);
    compute_sphere_vertices_functor func_vt(resolution, radius);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_vertices - 2),
                      mesh_ptr->vertices_.begin() + 2, func_vt);
    mesh_ptr->triangles_.resize(4 * resolution +
                                4 * (resolution - 2) * resolution);
    compute_sphere_triangles_functor1 func_tr1(
            thrust::raw_pointer_cast(mesh_ptr->triangles_.data()), resolution);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(2 * resolution),
                     func_tr1);
    compute_sphere_triangles_functor2 func_tr2(
            thrust::raw_pointer_cast(mesh_ptr->triangles_.data()) +
                    4 * resolution,
            resolution);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(
                             2 * (resolution - 1) * resolution),
                     func_tr2);
    return mesh_ptr;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateHalfSphere(
        float radius /* = 1.0*/, int resolution /* = 20*/) {
    auto mesh_ptr = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateHalfSphere] radius <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateHalfSphere] resolution <= 0");
    }
    size_t n_vertices = resolution * resolution + 1;
    mesh_ptr->vertices_.resize(n_vertices);
    mesh_ptr->vertices_[0] = Eigen::Vector3f(0.0, 0.0, radius);
    compute_sphere_vertices_functor func_vt(resolution, radius);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_vertices - 1),
                      mesh_ptr->vertices_.begin() + 1, func_vt);
    mesh_ptr->triangles_.resize(2 * resolution +
                                4 * (resolution / 2 - 1) * resolution);
    compute_half_sphere_triangles_functor1 func_tr1(
            thrust::raw_pointer_cast(mesh_ptr->triangles_.data()), resolution);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(2 * resolution),
                     func_tr1);
    compute_sphere_triangles_functor2 func_tr2(
            thrust::raw_pointer_cast(mesh_ptr->triangles_.data()) +
                    2 * resolution,
            resolution, 1);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(
                             2 * (resolution / 2 - 1) * resolution),
                     func_tr2);
    return mesh_ptr;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateCylinder(
        float radius /* = 1.0*/,
        float height /* = 2.0*/,
        int resolution /* = 20*/,
        int split /* = 4*/) {
    auto mesh_ptr = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateCylinder] radius <= 0");
    }
    if (height <= 0) {
        utility::LogError("[CreateCylinder] height <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateCylinder] resolution <= 0");
    }
    if (split <= 0) {
        utility::LogError("[CreateCylinder] split <= 0");
    }
    size_t n_vertices = resolution * (split + 1) + 2;
    mesh_ptr->vertices_.resize(n_vertices);
    mesh_ptr->vertices_[0] = Eigen::Vector3f(0.0, 0.0, height * 0.5);
    mesh_ptr->vertices_[1] = Eigen::Vector3f(0.0, 0.0, -height * 0.5);
    float step = M_PI * 2.0 / (float)resolution;
    float h_step = height / (float)split;
    compute_cylinder_vertices_functor func_vt(resolution, radius, height, step,
                                              h_step);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator<size_t>(n_vertices - 2),
                      mesh_ptr->vertices_.begin() + 2, func_vt);
    mesh_ptr->triangles_.resize(2 * resolution + 2 * split * resolution);
    compute_cylinder_triangles_functor1 func_tr1(
            thrust::raw_pointer_cast(mesh_ptr->triangles_.data()), resolution,
            split);
    for_each(thrust::make_counting_iterator<size_t>(0),
             thrust::make_counting_iterator<size_t>(resolution), func_tr1);
    compute_cylinder_triangles_functor2 func_tr2(
            thrust::raw_pointer_cast(mesh_ptr->triangles_.data()) + 2 * resolution,
            resolution);
    for_each(thrust::make_counting_iterator<size_t>(0),
             thrust::make_counting_iterator<size_t>(resolution * split),
             func_tr2);
    return mesh_ptr;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateTube(
        float radius /* = 1.0*/,
        float height /* = 2.0*/,
        int resolution /* = 20*/,
        int split /* = 4*/) {
    auto mesh_ptr = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateTube] radius <= 0");
    }
    if (height <= 0) {
        utility::LogError("[CreateTube] height <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateTube] resolution <= 0");
    }
    if (split <= 0) {
        utility::LogError("[CreateTube] split <= 0");
    }
    size_t n_vertices = resolution * (split + 1);
    mesh_ptr->vertices_.resize(n_vertices);
    float step = M_PI * 2.0 / (float)resolution;
    float h_step = height / (float)split;
    compute_cylinder_vertices_functor func_vt(resolution, radius, height, step,
                                              h_step);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator<size_t>(n_vertices),
                      mesh_ptr->vertices_.begin(), func_vt);
    mesh_ptr->triangles_.resize(2 * split * resolution);
    compute_cylinder_triangles_functor2 func_tr2(
            thrust::raw_pointer_cast(mesh_ptr->triangles_.data()),
            resolution, 0);
    for_each(thrust::make_counting_iterator<size_t>(0),
             thrust::make_counting_iterator<size_t>(resolution * split),
             func_tr2);
    return mesh_ptr;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateCapsule(
        float radius /* = 1.0*/,
        float height /* = 2.0*/,
        int resolution /* = 20*/,
        int split /* = 4*/) {
    auto mesh_ptr = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateCapsule] radius <= 0");
    }
    if (height <= 0) {
        utility::LogError("[CreateCapsule] height <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateCapsule] resolution <= 0");
    }
    if (split <= 0) {
        utility::LogError("[CreateCapsule] split <= 0");
    }
    Eigen::Matrix4f transform;
    auto mesh_top = CreateHalfSphere(radius, resolution);
    transform << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, height / 2.0, 0, 0, 0, 1;
    mesh_top->Transform(transform);
    auto mesh_bottom = CreateHalfSphere(radius, resolution);
    transform << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, -height / 2.0, 0, 0, 0, 1;
    mesh_bottom->Transform(transform);
    mesh_ptr = CreateTube(radius, height, resolution, split);
    *mesh_ptr += *mesh_top;
    *mesh_ptr += *mesh_bottom;
    return mesh_ptr;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateCone(float radius /* = 1.0*/,
                                                       float height /* = 2.0*/,
                                                       int resolution /* = 20*/,
                                                       int split /* = 4*/) {
    auto mesh_ptr = std::make_shared<TriangleMesh>();
    if (radius <= 0) {
        utility::LogError("[CreateCone] radius <= 0");
    }
    if (height <= 0) {
        utility::LogError("[CreateCone] height <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateCone] resolution <= 0");
    }
    if (split <= 0) {
        utility::LogError("[CreateCone] split <= 0");
    }
    mesh_ptr->vertices_.resize(resolution * split + 2);
    mesh_ptr->vertices_[0] = Eigen::Vector3f(0.0, 0.0, 0.0);
    mesh_ptr->vertices_[1] = Eigen::Vector3f(0.0, 0.0, height);
    float step = M_PI * 2.0 / (float)resolution;
    float h_step = height / (float)split;
    float r_step = radius / (float)split;
    compute_cone_vertices_functor func_vt(resolution, split, step, r_step,
                                          h_step);
    thrust::transform(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(resolution * split),
            mesh_ptr->vertices_.begin() + 2, func_vt);
    mesh_ptr->triangles_.resize(2 * resolution + 2 * (split - 1) * resolution);
    compute_cone_triangles_functor1 func_tr1(
            thrust::raw_pointer_cast(mesh_ptr->triangles_.data()), resolution,
            split);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(resolution),
                     func_tr1);
    compute_cone_triangles_functor2 func_tr2(
            thrust::raw_pointer_cast(mesh_ptr->triangles_.data()) + 2 * resolution,
            resolution);
    thrust::for_each(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>((split - 1) * resolution),
            func_tr2);
    return mesh_ptr;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateTorus(
        float torus_radius /* = 1.0 */,
        float tube_radius /* = 0.5 */,
        int radial_resolution /* = 20 */,
        int tubular_resolution /* = 20 */) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (torus_radius <= 0) {
        utility::LogError("[CreateTorus] torus_radius <= 0");
    }
    if (tube_radius <= 0) {
        utility::LogError("[CreateTorus] tube_radius <= 0");
    }
    if (radial_resolution <= 0) {
        utility::LogError("[CreateTorus] radial_resolution <= 0");
    }
    if (tubular_resolution <= 0) {
        utility::LogError("[CreateTorus] tubular_resolution <= 0");
    }

    mesh->vertices_.resize(radial_resolution * tubular_resolution);
    mesh->triangles_.resize(2 * radial_resolution * tubular_resolution);
    compute_torus_mesh_functor func(thrust::raw_pointer_cast(mesh->vertices_.data()),
                                    thrust::raw_pointer_cast(mesh->triangles_.data()),
                                    torus_radius, tube_radius,
                                    radial_resolution, tubular_resolution);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(radial_resolution * tubular_resolution), func);
    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateArrow(
        float cylinder_radius /* = 1.0*/,
        float cone_radius /* = 1.5*/,
        float cylinder_height /* = 5.0*/,
        float cone_height /* = 4.0*/,
        int resolution /* = 20*/,
        int cylinder_split /* = 4*/,
        int cone_split /* = 1*/) {
    if (cylinder_radius <= 0) {
        utility::LogError("[CreateArrow] cylinder_radius <= 0");
    }
    if (cone_radius <= 0) {
        utility::LogError("[CreateArrow] cone_radius <= 0");
    }
    if (cylinder_height <= 0) {
        utility::LogError("[CreateArrow] cylinder_height <= 0");
    }
    if (cone_height <= 0) {
        utility::LogError("[CreateArrow] cone_height <= 0");
    }
    if (resolution <= 0) {
        utility::LogError("[CreateArrow] resolution <= 0");
    }
    if (cylinder_split <= 0) {
        utility::LogError("[CreateArrow] cylinder_split <= 0");
    }
    if (cone_split <= 0) {
        utility::LogError("[CreateArrow] cone_split <= 0");
    }
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    auto mesh_cylinder = CreateCylinder(cylinder_radius, cylinder_height,
                                        resolution, cylinder_split);
    transformation(2, 3) = cylinder_height * 0.5;
    mesh_cylinder->Transform(transformation);
    auto mesh_cone =
            CreateCone(cone_radius, cone_height, resolution, cone_split);
    transformation(2, 3) = cylinder_height;
    mesh_cone->Transform(transformation);
    auto mesh_arrow = mesh_cylinder;
    *mesh_arrow += *mesh_cone;
    return mesh_arrow;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateCoordinateFrame(
        float size /* = 1.0*/,
        const Eigen::Vector3f &origin /* = Eigen::Vector3f(0.0, 0.0, 0.0)*/) {
    if (size <= 0) {
        utility::LogError("[CreateCoordinateFrame] size <= 0");
    }
    auto mesh_frame = CreateSphere(0.06 * size);
    mesh_frame->ComputeVertexNormals();
    mesh_frame->PaintUniformColor(Eigen::Vector3f(0.5, 0.5, 0.5));

    std::shared_ptr<TriangleMesh> mesh_arrow;
    Eigen::Matrix4f transformation;

    mesh_arrow = CreateArrow(0.035 * size, 0.06 * size, 0.8 * size, 0.2 * size);
    mesh_arrow->ComputeVertexNormals();
    mesh_arrow->PaintUniformColor(Eigen::Vector3f(1.0, 0.0, 0.0));
    transformation << 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    mesh_arrow->Transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_arrow = CreateArrow(0.035 * size, 0.06 * size, 0.8 * size, 0.2 * size);
    mesh_arrow->ComputeVertexNormals();
    mesh_arrow->PaintUniformColor(Eigen::Vector3f(0.0, 1.0, 0.0));
    transformation << 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1;
    mesh_arrow->Transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_arrow = CreateArrow(0.035 * size, 0.06 * size, 0.8 * size, 0.2 * size);
    mesh_arrow->ComputeVertexNormals();
    mesh_arrow->PaintUniformColor(Eigen::Vector3f(0.0, 0.0, 1.0));
    transformation << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    mesh_arrow->Transform(transformation);
    *mesh_frame += *mesh_arrow;

    transformation = Eigen::Matrix4f::Identity();
    transformation.block<3, 1>(0, 3) = origin;
    mesh_frame->Transform(transformation);

    return mesh_frame;
}

std::shared_ptr<TriangleMesh> TriangleMesh::CreateMoebius(
        int length_split /* = 70 */,
        int width_split /* = 15 */,
        int twists /* = 1 */,
        float radius /* = 1 */,
        float flatness /* = 1 */,
        float width /* = 1 */,
        float scale /* = 1 */) {
    auto mesh = std::make_shared<TriangleMesh>();
    if (length_split <= 0) {
        utility::LogError("[CreateMoebius] length_split <= 0");
    }
    if (width_split <= 0) {
        utility::LogError("[CreateMoebius] width_split <= 0");
    }
    if (twists < 0) {
        utility::LogError("[CreateMoebius] twists < 0");
    }
    if (radius <= 0) {
        utility::LogError("[CreateMoebius] radius <= 0");
    }
    if (flatness == 0) {
        utility::LogError("[CreateMoebius] flatness == 0");
    }
    if (width <= 0) {
        utility::LogError("[CreateMoebius] width <= 0");
    }
    if (scale <= 0) {
        utility::LogError("[CreateMoebius] scale <= 0");
    }

    mesh->vertices_.resize(length_split * width_split);
    compute_moebius_vertices_functor func1(length_split, width_split, twists, radius, flatness, width, scale);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator<size_t>(length_split * width_split),
                      mesh->vertices_.begin(), func1);

    mesh->triangles_.resize(2 * length_split * (width_split - 1));
    compute_moebius_triangles_functor func2(thrust::raw_pointer_cast(mesh->triangles_.data()), length_split,
                                            width_split, twists);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(length_split * (width_split - 1)), func2);
    return mesh;
}