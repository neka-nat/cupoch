<p align="center">
<img src="https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/cupoch_logo.png" width="320" />
</p>

# CUDA-based 3D Data Processing Library

[![Build Status](https://travis-ci.com/neka-nat/cupoch.svg?branch=master)](https://travis-ci.com/neka-nat/cupoch)
[![PyPI version](https://badge.fury.io/py/cupoch.svg)](https://badge.fury.io/py/cupoch)
[![Downloads](https://pepy.tech/badge/cupoch)](https://pepy.tech/project/cupoch)

Cupoch is a library that implements rapid 3D data processing using CUDA.

## Core Features

* 3D data processing using CUDA
* [Open3D](https://github.com/intel-isl/Open3D)-like API
* Interactive GUI (OpenGL CUDA interop and [imgui](https://github.com/ocornut/imgui))
* Interoperability between cupoch 3D data and [DLPack](https://github.com/dmlc/dlpack)(Pytorch, Cupy,...) data structure

## Installation

This software is tested under 64 Bit Ubuntu Linux 18.04 and CUDA 10.0.
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

![speedup](https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/speedup.png)