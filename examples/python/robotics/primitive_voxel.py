import cupoch as cph
import numpy as np

sphere = cph.collision.Sphere(1.0)
cph.visualization.draw_geometries([sphere.create_voxel_grid(0.1)])
cph.visualization.draw_geometries([sphere.create_voxel_grid_with_sweeping(0.1,
    np.array([[1.0, 0.0, 0.0, 1.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]]), 10)])