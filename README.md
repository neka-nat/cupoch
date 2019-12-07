<p align="center">
<img src="docs/_static/cupoch_logo.png" width="320" />
</p>

# cupoch

Cupoch is a library that implements rapid 3D data processing using CUDA.

## Core Features

* 3D data processing using CUDA
* [Open3D](https://github.com/intel-isl/Open3D)-like API
* OpenGL/DirectX-based GUI (not using any GUI frameworks, such as Qt, wxwidget...) 
* Interoperability between cupoch 3D data and [DLPack](https://github.com/dmlc/dlpack.git) data structure

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