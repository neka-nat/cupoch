import numpy as np
import cupoch as cph

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../misc"))
import meshes


def test_mesh(noise=0):
    mesh = meshes.knot()
    if noise > 0:
        vertices = np.asarray(mesh.vertices.cpu())
        vertices += np.random.uniform(0, noise, size=vertices.shape)
        mesh.vertices = cph.utility.Vector3fVector(vertices)
    mesh.compute_vertex_normals()
    return mesh


if __name__ == "__main__":
    in_mesh = test_mesh()
    cph.visualization.draw_geometries([in_mesh])

    mesh = in_mesh.filter_sharpen(number_of_iterations=1, strength=1)
    cph.visualization.draw_geometries([mesh])

    in_mesh = test_mesh(noise=5)
    cph.visualization.draw_geometries([in_mesh])

    mesh = in_mesh.filter_smooth_simple(number_of_iterations=1)
    cph.visualization.draw_geometries([mesh])

    cph.visualization.draw_geometries([mesh])
    mesh = in_mesh.filter_smooth_laplacian(number_of_iterations=100)
    cph.visualization.draw_geometries([mesh])

    cph.visualization.draw_geometries([mesh])
    mesh = in_mesh.filter_smooth_taubin(number_of_iterations=100)
    cph.visualization.draw_geometries([mesh])
