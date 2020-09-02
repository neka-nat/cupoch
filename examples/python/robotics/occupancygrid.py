import cupoch as cph
import numpy as np

x = np.linspace(0.0, 1.0, 100)
y = np.linspace(0.0, 1.0, 100)
X, Y = np.meshgrid(x, y)
Z = np.ones((100, 100))
plane_grid = np.stack([X, Y, Z], axis=2).reshape(-1, 3)

pcd = cph.geometry.PointCloud()
pcd.points = cph.utility.Vector3fVector(plane_grid)
cph.visualization.draw_geometries([pcd])

ocg = cph.geometry.OccupancyGrid()
ocg.insert(pcd, np.zeros(3))
print(ocg)
cph.visualization.draw_geometries([ocg])
ocg.visualize_free_area = False
cph.visualization.draw_geometries([ocg])

pcd = cph.geometry.PointCloud.create_from_occupancygrid(ocg)
cph.visualization.draw_geometries([pcd])