import numpy as np
import cupoch as cph
import matplotlib.pyplot as plt

np.random.seed(42)


def pointcloud_generator():
    yield "cube", cph.geometry.TriangleMesh.create_sphere(
    ).sample_points_uniformly(int(1e4)), 0.4

    mesh = cph.geometry.TriangleMesh.create_torus().scale(5)
    mesh += cph.geometry.TriangleMesh.create_torus()
    yield "torus", mesh.sample_points_uniformly(int(1e4)), 0.75

    d = 4
    mesh = cph.geometry.TriangleMesh.create_tetrahedron().translate((-d, 0, 0))
    mesh += cph.geometry.TriangleMesh.create_octahedron().translate((0, 0, 0))
    mesh += cph.geometry.TriangleMesh.create_icosahedron().translate((d, 0, 0))
    mesh += cph.geometry.TriangleMesh.create_torus().translate((-d, -d, 0))
    mesh += cph.geometry.TriangleMesh.create_moebius(twists=1).translate(
        (0, -d, 0))
    mesh += cph.geometry.TriangleMesh.create_moebius(twists=2).translate(
        (d, -d, 0))
    yield "shapes", mesh.sample_points_uniformly(int(1e5)), 0.5

    yield "fragment", cph.io.read_point_cloud(
        "../../testdata/fragment.ply"), 0.02


if __name__ == "__main__":
    cph.utility.set_verbosity_level(cph.utility.Debug)

    cmap = plt.get_cmap("tab20")
    for pcl_name, pcl, eps in pointcloud_generator():
        print("%s has %d points" % (pcl_name, np.asarray(pcl.points.cpu()).shape[0]))
        cph.visualization.draw_geometries([pcl])

        labels = np.array(
            pcl.cluster_dbscan(eps=eps, min_points=10, print_progress=True))
        max_label = labels.max()
        print("%s has %d clusters" % (pcl_name, max_label + 1))

        colors = cmap(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcl.colors = cph.utility.Vector3fVector(colors[:, :3])
        cph.visualization.draw_geometries([pcl])