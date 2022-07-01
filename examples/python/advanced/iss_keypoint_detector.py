import os
import sys
import time
import cupoch as cph

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../misc"))
import meshes

# Compute ISS Keypoints on Armadillo
mesh = meshes.armadillo()
pcd = cph.geometry.PointCloud()
pcd.points = mesh.vertices

tic = time.time()
keypoints, masks = cph.geometry.keypoint.compute_iss_keypoints(pcd)
toc = 1000 * (time.time() - tic)
print("ISS Computation took {:.0f} [ms]".format(toc))
print(len(pcd.points), len(keypoints.points))

mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.5, 0.5, 0.5])
keypoints.paint_uniform_color([1.0, 0.75, 0.0])
cph.visualization.draw_geometries([keypoints, mesh])
