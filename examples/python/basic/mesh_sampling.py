import numpy as np
import cupoch as cph

import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../misc'))
import meshes


if __name__ == "__main__":
    plane = meshes.plane()
    cph.visualization.draw_geometries([plane])

    print('Uniform sampling can yield clusters of points on the surface')
    pcd = plane.sample_points_uniformly(number_of_points=500)
    cph.visualization.draw_geometries([pcd])
