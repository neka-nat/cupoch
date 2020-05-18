import cupoch as cph
import numpy as np

sphere = cph.collision.Sphere(1.0)
cph.visualization.draw_geometries([cph.collision.create_voxel_grid(sphere, 0.1)])
cph.visualization.draw_geometries([cph.collision.create_voxel_grid_with_sweeping(sphere, 0.1,
    np.array([[1.0, 0.0, 0.0, 1.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]]), 10)])