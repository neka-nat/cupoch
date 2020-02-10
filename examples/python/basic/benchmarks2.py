import os
os.environ["OMP_NUM_THREADS"] = str(1)
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


pinhole_camera_intrinsic_cpu = o3d.io.read_pinhole_camera_intrinsic(
    "../../testdata/camera_primesense.json")
source_color_cpu = o3d.io.read_image("../../testdata/rgbd/color/00000.jpg")
source_depth_cpu = o3d.io.read_image("../../testdata/rgbd/depth/00000.png")
target_color_cpu = o3d.io.read_image("../../testdata/rgbd/color/00001.jpg")
target_depth_cpu = o3d.io.read_image("../../testdata/rgbd/depth/00001.png")
print(source_color_cpu)
pinhole_camera_intrinsic_gpu = cph.io.read_pinhole_camera_intrinsic(
    "../../testdata/camera_primesense.json")
source_color_gpu = cph.io.read_image("../../testdata/rgbd/color/00000.jpg")
source_depth_gpu = cph.io.read_image("../../testdata/rgbd/depth/00000.png")
target_color_gpu = cph.io.read_image("../../testdata/rgbd/color/00001.jpg")
target_depth_gpu = cph.io.read_image("../../testdata/rgbd/depth/00001.png")
print(source_color_gpu)

speeds = {}

source_rgbd_image_cpu = o3d.geometry.RGBDImage.create_from_color_and_depth(
    source_color_cpu, source_depth_cpu)
target_rgbd_image_cpu = o3d.geometry.RGBDImage.create_from_color_and_depth(
    target_color_cpu, target_depth_cpu)
target_pcd_cpu = o3d.geometry.PointCloud.create_from_rgbd_image(
    target_rgbd_image_cpu, pinhole_camera_intrinsic_cpu)
option_cpu = o3d.odometry.OdometryOption()

source_rgbd_image_gpu = cph.geometry.RGBDImage.create_from_color_and_depth(
    source_color_gpu, source_depth_gpu)
target_rgbd_image_gpu = cph.geometry.RGBDImage.create_from_color_and_depth(
    target_color_gpu, target_depth_gpu)
target_pcd_gpu = cph.geometry.PointCloud.create_from_rgbd_image(
    target_rgbd_image_gpu, pinhole_camera_intrinsic_gpu)
option_gpu = cph.odometry.OdometryOption()

odo_init = np.identity(4)

_, tc = measure_time(o3d.odometry, "compute_rgbd_odometry", "CPU",
                     source_rgbd_image_cpu, target_rgbd_image_cpu, pinhole_camera_intrinsic_cpu,
                     odo_init, o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option_cpu)
_, tg = measure_time(cph.odometry, "compute_rgbd_odometry", "GPU",
                     source_rgbd_image_gpu, target_rgbd_image_gpu, pinhole_camera_intrinsic_gpu,
                     odo_init, cph.odometry.RGBDOdometryJacobianFromHybridTerm(), option_gpu)
speeds['compute_rgbd_odometry'] = (tc / tg)


import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.title("Speedup over open3d (%d pixel image)" % (source_color_gpu.width * source_color_gpu.height))
plt.ylabel("Speedup")
plt.bar(speeds.keys(), speeds.values())
plt.xticks(rotation=70)
plt.tight_layout()
plt.show()