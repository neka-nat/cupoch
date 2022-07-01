import cupoch as cph
from global_registration import *
import numpy as np
import copy

import time


def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result = cph.registration.registration_fast_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        cph.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold),
    )
    return result


if __name__ == "__main__":
    cph.utility.set_verbosity_level(cph.utility.Debug)

    voxel_size = 0.05  # means 5cm for the dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

    start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    print(result_fast.transformation)
    draw_registration_result(source_down, target_down, result_fast.transformation)
