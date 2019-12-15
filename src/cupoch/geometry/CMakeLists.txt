cuda_add_library(cupoch_geometry pointcloud.cu
                 estimate_normals.cu
                 down_sample.cu
                 geometry3d.cu
                 kdtree_flann.cu)
target_link_libraries(cupoch_geometry cupoch_utility flann_cuda_s)