<p align="center">
<img src="https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/cupoch_logo.png" width="320" />
</p>

# Robotics with GPU computing

[![Build Status](https://travis-ci.com/neka-nat/cupoch.svg?branch=master)](https://travis-ci.com/neka-nat/cupoch)
[![PyPI version](https://badge.fury.io/py/cupoch.svg)](https://badge.fury.io/py/cupoch)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cupoch)
[![Downloads](https://pepy.tech/badge/cupoch)](https://pepy.tech/project/cupoch)

Cupoch is a library that implements rapid 3D data processing and robotics computation using CUDA.

The goal of this library is to process the input of 3D sensors rapidly and use it to control the robot.
This library is based on the functionality of Open3D, with additional features for integration into robot systems.

## Core Features

* 3D data processing and robotics computation using CUDA
    * Point cloud registration
        * ICP
        * [Colored Point Cloud Registration](https://ieeexplore.ieee.org/document/8237287)
        * [Fast Global Registration](http://vladlen.info/papers/fast-global-registration.pdf)
        * [FilterReg](https://arxiv.org/abs/1811.10136)
    * Point cloud clustering
        * [G-DBSCAN: A GPU Accelerated Algorithm for Density-based Clustering](https://www.sciencedirect.com/science/article/pii/S1877050913003438)
    * Point cloud/Triangle mesh filtering, down sampling
    * Visual Odometry
        * [Real-time visual odometry from dense RGB-D images](https://ieeexplore.ieee.org/document/6130321)
        * [Robust Odometry Estimation for RGB-D Cameras](https://ieeexplore.ieee.org/document/6631104)
    * Collision checking
    * Occupancy grid
    * Distance transform
        * [Parallel Banding Algorithm to Compute Exact Distance Transform with the GPU](https://www.comp.nus.edu.sg/~tants/pba.html)
    * Path finding on graph structure
    * Path planning for collision avoidance
* [Open3D](https://github.com/intel-isl/Open3D)-like API
* Support memory pool and managed allocators
* Interactive GUI (OpenGL CUDA interop and [imgui](https://github.com/ocornut/imgui))
* Interoperability between cupoch 3D data and [DLPack](https://github.com/dmlc/dlpack)(Pytorch, Cupy,...) data structure

## Installation

This software is tested under 64 Bit Ubuntu Linux 18.04 and CUDA 10.0/10.1/10.2.
You can install cupoch using pip.

```
pip install cupoch
```

Or install cupoch from source.

```
git clone https://github.com/neka-nat/cupoch.git --recurse
cd cupoch
mkdir build
cd build
cmake ..; make install-pip-package -j
```

### Installation for Jetson
You can also install cupoch using pip on Jetson.
Please set up Jetson using [jetcard](https://github.com/NVIDIA-AI-IOT/jetcard) and install some packages with apt.

```
sudo apt-get install libxinerama-dev libxcursor-dev libglu1-mesa-dev
pip3 install https://github.com/neka-nat/cupoch/releases/download/v0.1.0/cupoch-0.1.0.0-cp36-cp36m-linux_aarch64.whl
```

## Results
The figure shows Cupoch's point cloud algorithms speedup over Open3D.
The environment tested on has the following specs:
* Intel Core i7-7700HQ CPU
* Nvidia GTX1070 GPU
* OMP_NUM_THREAD=1

You can get the result by running the example script in your environment.

```
cd examples/python/basic
python benchmarks.py
```

![speedup](https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/speedup.png)

### Visual odometry with intel realsense D435

![vo](https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/vo_gpu.gif)

### Occupancy grid with intel realsense D435

![og](https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/og_gpu.gif)

### Fast Global Registration

![fgr](https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/fgr.png)

### Path finding

![pf](https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/path_finding.png)

### Visual odometry with ROS + D435

This demo works in the following environment.
* ROS melodic
* Python2.7

```
# Launch roscore and rviz in the other terminals.
cd examples/python/ros
python realsense_rgbd_odometry_node.py
```

![vo](https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/ros_vo.gif)

## Visualization

| Point Cloud | Triangle Mesh | Voxel Grid |
|-------------|---------------|------------|
| <img src="https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/pointcloud.png" width="640"> |  <img src="https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/trianglemesh.png" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/voxelgrid.png" width="640"> |

| Occupancy Grid | Graph | Image |
|----------------|-------|-------|
| <img src="https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/occupancygrid.png" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/graph.png" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/image.png" width="640"> |

## References

* CUDA repository forked from Open3D, https://github.com/theNded/Open3D
* GPU computing in Robotics, https://github.com/JanuszBedkowski/gpu_computing_in_robotics
* Voxel collision comupation for robotics, https://github.com/fzi-forschungszentrum-informatik/gpu-voxels