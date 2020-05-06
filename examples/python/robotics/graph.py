import os
import itertools
import time
import cupoch as cph
import numpy as np
import networkx as nx

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../misc'))
import meshes

n_nodes = 50
points = np.random.rand(n_nodes, 3)
gp = cph.geometry.Graph()
gp.points = cph.utility.Vector3fVector(points)

for c in itertools.combinations(list(range(n_nodes)), 2):
    gp.add_edge(c)
gp.set_edge_weights_from_distance()

gp.remove_edge((0, 30))

path = gp.dijkstra_path(0, 30)
print("Find path: ", path)
for i in range(len(path[:-1])):
    gp.paint_node_color(path[i], (0.0, 1.0, 0.0))
    gp.paint_edge_color((path[i], path[i+1]), (1.0, 0.0, 0.0))
gp.paint_node_color(path[-1], (0.0, 1.0, 0.0))
cph.visualization.draw_geometries([gp])


# Graph from triangle mesh
mesh = meshes.bunny()
mesh.remove_unreferenced_vertices()
gp = cph.geometry.Graph.create_from_triangle_mesh(mesh)
gp.set_edge_weights_from_distance()
start = time.time()
path = gp.dijkstra_path(0, 500)
elapsed_time = time.time() - start
print("Find path (GPU): ", path, " Time: ", elapsed_time)
for i in range(len(path[:-1])):
    gp.paint_node_color(path[i], (0.0, 1.0, 0.0))
    gp.paint_edge_color((path[i], path[i+1]), (1.0, 0.0, 0.0))
gp.paint_node_color(path[-1], (0.0, 1.0, 0.0))
cph.visualization.draw_geometries([gp])

h_edges = gp.edges.cpu()
h_weights = gp.edge_weights.cpu()
h_g = nx.Graph()
for e, w in zip(h_edges, h_weights):
    h_g.add_edge(e[0], e[1], weight=w)
start = time.time()
path = nx.dijkstra_path(h_g, 0, 500)
elapsed_time = time.time() - start
print("Find path (CPU, networkx): ", path, " Time: ", elapsed_time)