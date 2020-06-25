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

ply_file = "../../testdata/sphere.ply"
tr_cpu, tc = measure_time(o3d.io, "read_triangle_mesh", "CPU", ply_file)
print(tr_cpu)
tr_gpu, tg = measure_time(cph.io, "read_triangle_mesh", "GPU", ply_file)
print(tr_gpu)

speeds = {}

num_of_points = int(1e5)
_, tc = measure_time(tr_cpu, "sample_points_uniformly", "CPU", num_of_points)
_, tg = measure_time(tr_gpu, "sample_points_uniformly", "GPU", num_of_points)
speeds['transform'] = (tc / tg)

import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.title("Speedup over open3d (%d vertices mesh)" % np.asarray(tr_gpu.vertices.cpu()).shape[0])
plt.ylabel("Speedup")
plt.bar(speeds.keys(), speeds.values())
plt.xticks(rotation=70)
plt.tight_layout()
plt.show()