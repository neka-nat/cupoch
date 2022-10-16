# Getting Started

As an introduction, let's try to process the point cloud.
The point cloud file used as a sample is located `examples/testdata/fragment.pcd`.

The first step is to import `cupoch`.
We will name it `cph` for short and use it in the following.

```py
import cupoch as cph
```

Before we get into the actual program, let's check if CUDA is available.
If you get `True` when executing the following function, CUDA is available.

```py
cph.utility.is_cuda_available()
# True or False
```

If the check is `True`, try to read the point cloud file.

```py
pointcloud = cph.io.read_point_cloud("examples/testdata/fragment.pcd")
```

Once successfully loaded, try to display the loaded point cloud.
Use visualizer as follows.

```py
cph.visualization.draw_geometries([pointcloud])
```

If the following window appears, you have succeeded.

![getting_started_1](https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/getting_started_1.png)

Close the window and then downsample.
Call the function that performs downsampling as follows.

```py
downsampled = pointcloud.voxel_down_sample(0.05)
```

Draw the point cloud after downsampling.

![getting_started_2](https://raw.githubusercontent.com/neka-nat/cupoch/master/docs/_static/getting_started_2.png)

That concludes our introduction!
There's plenty of other features you can use and experiment with!
