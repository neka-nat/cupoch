import numpy as np
import cupoch as cph
import torch
from torch.utils.dlpack import to_dlpack

tensor = torch.rand(10, 3).cuda()
print("Torch tensor:")
print(tensor)
ts_dl = to_dlpack(tensor)
pcd = cph.geometry.PointCloud()
pcd.from_points_dlpack(ts_dl)
print("PointCloud.points:")
print(np.asarray(pcd.points.cpu()))

cpu_ts_dl = to_dlpack(tensor.cpu())
pcd.from_points_dlpack(cpu_ts_dl)
print("PointCloud.points:")
print(np.asarray(pcd.points.cpu()))
