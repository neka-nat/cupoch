import cupoch as cph
import numpy as np

pcd = cph.geometry.PointCloud()

print("---- From list of 1D np.array")
pts = [np.random.rand(3) for _ in range(10)]
pcd.points = cph.utility.Vector3fVector(pts)

print("---- From 2D np.array")
pts = np.random.rand(10, 3)
pcd.points = cph.utility.Vector3fVector(pts)

print("---- To 2D np.array")
print(np.asarray(pcd.points.cpu()))
