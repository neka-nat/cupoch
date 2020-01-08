<p align="center">
<img src="docs/_static/cupoch_logo.png" width="320" />
</p>

# CUDA-based 3D Data Processing Library

Cupoch is a library that implements rapid 3D data processing using CUDA.

## Core Features

* 3D data processing using CUDA
* [Open3D](https://github.com/intel-isl/Open3D)-like API
* Interactive GUI (using OpenGL CUDA interop and [imgui](https://github.com/ocornut/imgui))
* Interoperability between cupoch 3D data and [DLPack](https://github.com/dmlc/dlpack)(Pytorch, Cupy,...) data structure

## Installation

```
pip install cupoch
```

```
git clone https://github.com/neka-nat/cupoch.git --recurse
cd cupoch
mkdir build
cd build
cmake ..; make -j
```