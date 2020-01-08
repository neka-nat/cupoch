import numpy as np
import cupoch as cph
import open3d as o3d
import time


def measure_time(obj, method_name, device, *args):
    fn = getattr(obj, method_name)
    start = time.time()
    res = fn(*args)
    elapsed_time = time.time() - start
    print("%s (%s) [sec]:" % (method_name, device), elapsed_time)
    return res, elapsed_time


# pcd_file = "../../testdata/icp/cloud_bin_0.pcd"
pcd_file = "../../testdata/fragment.ply"
pc_cpu, tc = measure_time(o3d.io, "read_point_cloud", "CPU", pcd_file)
print(pc_cpu)
pc_gpu, tg = measure_time(cph.io, "read_point_cloud", "GPU", pcd_file)
print(pc_gpu)

speeds = {}

tf = np.identity(4)
_, tc = measure_time(pc_cpu, "transform", "CPU", tf)
_, tg = measure_time(pc_gpu, "transform", "GPU", tf)
speeds['transform'] = (tc / tg)

_, tc = measure_time(pc_cpu, "estimate_normals", "CPU")
_, tg = measure_time(pc_gpu, "estimate_normals", "GPU")
speeds['estimate_normals'] = (tc / tg)

_, tc = measure_time(pc_cpu, "voxel_down_sample", "CPU", 0.005)
_, tg = measure_time(pc_gpu, "voxel_down_sample", "GPU", 0.005)
speeds['voxel_down_sample'] = (tc / tg)

_, tc = measure_time(pc_cpu, "remove_radius_outlier", "CPU", 10, 0.5)
_, tg = measure_time(pc_gpu, "remove_radius_outlier", "GPU", 10, 0.5)
speeds['remove_radius_outlier'] = (tc / tg)

_, tc = measure_time(pc_cpu, "remove_statistical_outlier", "CPU", 20, 2.0)
_, tg = measure_time(pc_gpu, "remove_statistical_outlier", "GPU", 20, 2.0)
speeds['remove_statistical_outlier'] = (tc / tg)

import matplotlib.pyplot as plt
plt.bar(speeds.keys(), speeds.values())
plt.show()