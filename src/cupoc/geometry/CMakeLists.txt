cuda_add_library(cupoc_geometry pointcloud.cu
                 estimate_normals.cu
                 down_sample.cu
                 geometry3d.cu
                 kdtree_flann.cu)
target_link_libraries(cupoc_geometry cupoc_utility flann_cuda)