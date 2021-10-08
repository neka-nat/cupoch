# Cupoch: Robotics with GPU computing

Cupoch is a library that implements rapid 3D data processing for robotics using CUDA.

The goal of this library is to implement fast 3D data computation in robot systems.
For example, it has applications in SLAM, collision avoidance, path planning and tracking.
This repository is based on [Open3D](https://github.com/intel-isl/Open3D).

## Core Features

* 3D data processing and robotics computation using CUDA
    * Point cloud registration
        * ICP
        * [Colored Point Cloud Registration](https://ieeexplore.ieee.org/document/8237287)
        * [Fast Global Registration](http://vladlen.info/papers/fast-global-registration.pdf)
        * [FilterReg](https://arxiv.org/abs/1811.10136)
    * Point cloud features
        * FPFH
        * SHOT
    * Point cloud clustering
        * [G-DBSCAN: A GPU Accelerated Algorithm for Density-based Clustering](https://www.sciencedirect.com/science/article/pii/S1877050913003438)
    * Point cloud/Triangle mesh filtering, down sampling
    * IO
        * Several file types(pcd, ply, stl, obj, urdf)
        * ROS message
    * Create Point Cloud from Laser Scan or RGBD Image
    * Visual Odometry
        * [Real-time visual odometry from dense RGB-D images](https://ieeexplore.ieee.org/document/6130321)
        * [Robust Odometry Estimation for RGB-D Cameras](https://ieeexplore.ieee.org/document/6631104)
    * Kinect Fusion
    * Stereo Matching
    * Collision checking
    * Occupancy grid
    * Distance transform
        * [Parallel Banding Algorithm to Compute Exact Distance Transform with the GPU](https://www.comp.nus.edu.sg/~tants/pba.html)
    * Path finding on graph structure
    * Path planning for collision avoidance
* Support memory pool and managed allocators
* Interactive GUI (OpenGL CUDA interop and [imgui](https://github.com/ocornut/imgui))
* Interoperability between cupoch 3D data and [DLPack](https://github.com/dmlc/dlpack)(Pytorch, Cupy,...) data structure
