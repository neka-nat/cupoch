<p align="center">
<img src="https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/cupoch_logo.png" width="320" />
</p>

# CUDA-based 3D Data Processing Library

[![Build Status](https://travis-ci.com/neka-nat/cupoch.svg?branch=master)](https://travis-ci.com/neka-nat/cupoch)
[![PyPI version](https://badge.fury.io/py/cupoch.svg)](https://badge.fury.io/py/cupoch)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cupoch)
[![Downloads](https://pepy.tech/badge/cupoch)](https://pepy.tech/project/cupoch)

Cupoch is a library that implements rapid 3D data processing using CUDA.

## Core Features

* 3D data processing using CUDA
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
pip3 install https://github.com/neka-nat/cupoch/releases/download/v0.0.7/cupoch-0.0.7.0-cp36-cp36m-linux_aarch64.whl
```

## Getting started
The following demo shows ICP registration.
You can write code that is almost compatible with open3d.

```py
import cupoch as cph
import numpy as np

if __name__ == "__main__":
    source_gpu = cph.io.read_point_cloud("testdata/icp/cloud_bin_0.pcd")
    target_gpu = cph.io.read_point_cloud("testdata/icp/cloud_bin_1.pcd")
    threshold = 0.02
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4],
                             [0.0, 0.0, 0.0, 1.0]])
    reg_p2p = cph.registration.registration_icp(
        source_gpu, target_gpu, threshold, trans_init.astype(np.float32),
        cph.registration.TransformationEstimationPointToPoint())
    print(reg_p2p.transformation)
    source_gpu.transform(reg_p2p.transformation)
    cph.visualization.draw_geometries([source_gpu, target_gpu])
```

## Result
The figure shows Cupoch speedup over Open3d.
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

| Point Cloud | Triangle Mesh | Voxel Grid | Image |
|-----|-----|---------|-----------|
| <img src="https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/pointcloud.png" width="640"> |  <img src="https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/trianglemesh.png" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/voxelgrid.png" width="640"> | <img src="https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/image.png" width="640"> |

## References

* CUDA repository forked from Open3D, https://github.com/theNded/Open3D