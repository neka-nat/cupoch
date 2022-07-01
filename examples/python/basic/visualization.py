import numpy as np
import cupoch as cph

if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    pcd = cph.io.read_point_cloud("../../testdata/fragment.ply")
    pcd.estimate_normals()
    cph.visualization.draw_geometries([pcd])

    print("Let's draw some primitives")
    mesh_box = cph.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_sphere = cph.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    mesh_cylinder = cph.geometry.TriangleMesh.create_capsule(radius=0.3, height=4.0)
    mesh_cylinder.compute_vertex_normals()
    mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
    mesh_frame = cph.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[-2, -2, -2])

    print("We draw a few primitives using collection.")
    cph.visualization.draw_geometries([mesh_box, mesh_sphere, mesh_cylinder, mesh_frame])

    print("We draw a few primitives using + operator of mesh.")
    cph.visualization.draw_geometries([mesh_box + mesh_sphere + mesh_cylinder + mesh_frame])

    print("Let's draw a cubic using o3d.geometry.LineSet.")
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = cph.geometry.LineSet(
        points=cph.utility.Vector3fVector(points),
        lines=cph.utility.Vector2iVector(lines),
    )
    line_set.colors = cph.utility.Vector3fVector(colors)
    cph.visualization.draw_geometries([line_set])

    print("Let's draw a textured triangle mesh from obj file.")
    textured_mesh = cph.io.read_triangle_mesh("../../testdata/crate/crate.obj")
    textured_mesh.compute_vertex_normals()
    cph.visualization.draw_geometries([textured_mesh])
