import numpy as np
import cupoch as cph
import open3d as o3d

pcd_file = "../../testdata/icp/cloud_bin_0.pcd"
pc_cpu = o3d.io.read_point_cloud(pcd_file)
print(pc_cpu)
pc_gpu = cph.io.read_point_cloud(pcd_file)
print(pc_gpu)