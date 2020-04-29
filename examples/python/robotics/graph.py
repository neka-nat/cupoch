import itertools
import cupoch as cph
import numpy as np

n_nodes = 50
points = np.random.rand(n_nodes, 3)
gp = cph.geometry.Graph()
gp.points = cph.utility.Vector3fVector(points)

for c in itertools.combinations(list(range(n_nodes)), 2):
    gp.add_edge(c)

cph.visualization.draw_geometries([gp])