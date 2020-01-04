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
    return res


# pcd_file = "../../testdata/icp/cloud_bin_0.pcd"
pcd_file = "../../testdata/fragment.ply"
pc_cpu = measure_time(o3d.io, "read_point_cloud", "CPU", pcd_file)
print(pc_cpu)
pc_gpu = measure_time(cph.io, "read_point_cloud", "GPU", pcd_file)
print(pc_gpu)

tf = np.identity(4)
measure_time(pc_cpu, "transform", "CPU", tf)
measure_time(pc_gpu, "transform", "GPU", tf)

measure_time(pc_cpu, "estimate_normals", "CPU")
measure_time(pc_gpu, "estimate_normals", "GPU")

measure_time(pc_cpu, "voxel_down_sample", "CPU", 0.005)
measure_time(pc_gpu, "voxel_down_sample", "GPU", 0.005)

measure_time(pc_cpu, "remove_statistical_outlier", "CPU", 20, 2.0)
measure_time(pc_gpu, "remove_statistical_outlier", "GPU", 20, 2.0)
