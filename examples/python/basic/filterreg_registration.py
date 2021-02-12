import cupoch as cph
import numpy as np
import time

if __name__ == "__main__":
    source_gpu = cph.io.read_point_cloud("../../testdata/icp/cloud_bin_0.pcd")
    target_gpu = cph.io.read_point_cloud("../../testdata/icp/cloud_bin_1.pcd")
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4],
                             [0.0, 0.0, 0.0, 1.0]])

    start = time.time()
    reg_p2p = cph.registration.registration_filterreg(
        source_gpu, target_gpu, trans_init.astype(np.float32))
    elapsed_time = time.time() - start
    print(reg_p2p.transformation)
    print("FilterReg (GPU) [sec]:", elapsed_time)
    source_gpu.transform(reg_p2p.transformation)
    cph.visualization.draw_geometries([source_gpu, target_gpu])