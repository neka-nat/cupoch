import os
os.environ["OMP_NUM_THREADS"] = str(1)
import numpy as np
import cupoch as cph
cph.initialize_allocator(cph.PoolAllocation, 1000000000)
import open3d as o3d
import time


def measure_time(obj, method_name, device, *args):
    fn = getattr(obj, method_name)
    start = time.time()
    res = fn(*args)
    elapsed_time = time.time() - start
    print("%s (%s) [sec]:" % (method_name, device), elapsed_time)
    return res, elapsed_time


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

_, tc = measure_time(pc_cpu, "remove_radius_outlier", "CPU", 10, 0.1)
_, tg = measure_time(pc_gpu, "remove_radius_outlier", "GPU", 10, 0.1)
speeds['remove_radius_outlier'] = (tc / tg)

_, tc = measure_time(pc_cpu, "remove_statistical_outlier", "CPU", 20, 2.0)
_, tg = measure_time(pc_gpu, "remove_statistical_outlier", "GPU", 20, 2.0)
speeds['remove_statistical_outlier'] = (tc / tg)

trans_init = np.asarray([[np.cos(np.deg2rad(30.0)), -np.sin(np.deg2rad(30.0)), 0.0, 0.0],
                         [np.sin(np.deg2rad(30.0)), np.cos(np.deg2rad(30.0)), 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])
tg_cpu = pc_cpu
tg_cpu.transform(trans_init)
tg_gpu = pc_gpu
tg_gpu.transform(trans_init)
threshold = 0.02
_, tc = measure_time(o3d.registration, "registration_icp", "CPU",
                     pc_cpu, tg_cpu, threshold, trans_init,
                     o3d.registration.TransformationEstimationPointToPoint())
_, tg = measure_time(cph.registration, "registration_icp", "GPU",
                     pc_gpu, tg_gpu, threshold, trans_init.astype(np.float32),
                     cph.registration.TransformationEstimationPointToPoint())
speeds['registration_icp'] = (tc / tg)

_, tc = measure_time(pc_cpu, "cluster_dbscan", "CPU", 0.02, 10)
_, tg = measure_time(pc_gpu, "cluster_dbscan", "GPU", 0.02, 10)
speeds['cluster_dbscan'] = (tc / tg)

import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.title("Speedup over CPU (%d points)" % np.asarray(pc_gpu.points.cpu()).shape[0])
plt.ylabel("Speedup")
plt.bar(speeds.keys(), speeds.values())
plt.xticks(rotation=70)
plt.tight_layout()
plt.show()