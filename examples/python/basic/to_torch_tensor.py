import numpy as np
import cupoch as cph
from torch.utils.dlpack import from_dlpack

pcd = cph.geometry.PointCloud()
pt = np.random.rand(10, 3)
print("Original:")
print(pt)
pcd.points = cph.utility.Vector3fVector(pt).cuda()
pt_dl = pcd.to_points_dlpack()
tensor = from_dlpack(pt_dl)
print("Torch tensor:")
print(tensor)