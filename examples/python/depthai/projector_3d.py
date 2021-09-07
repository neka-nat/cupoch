#!/usr/bin/env python3

# Code copied from main depthai repo, depthai_helpers/projector_3d.py

import numpy as np
import cupoch as cph

class PointCloudVisualizer():
    def __init__(self, intrinsic_matrix, width, height):
        self.depth_map = None
        self.rgb = None
        self.pcl = None

        self.pinhole_camera_intrinsic = cph.camera.PinholeCameraIntrinsic(width,
                                                                         height,
                                                                         intrinsic_matrix[0][0],
                                                                         intrinsic_matrix[1][1],
                                                                         intrinsic_matrix[0][2],
                                                                         intrinsic_matrix[1][2])
        self.vis = cph.visualization.Visualizer()
        self.vis.create_window()
        self.isstarted = False

    def rgbd_to_projection(self, depth_map, rgb, is_rgb):
        self.depth_map = depth_map
        self.rgb = rgb
        rgb_cph = cph.geometry.Image(self.rgb)
        depth_cph = cph.geometry.Image(self.depth_map)
        # TODO: query frame shape to get this, and remove the param 'is_rgb'
        if is_rgb:
            rgbd_image = cph.geometry.RGBDImage.create_from_color_and_depth(rgb_cph, depth_cph, convert_rgb_to_intensity=False)
        else:
            rgbd_image = cph.geometry.RGBDImage.create_from_color_and_depth(rgb_cph, depth_cph)
        if self.pcl is None:
            self.pcl = cph.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
        else:
            pcd = cph.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
            self.pcl.points = pcd.points
            self.pcl.colors = pcd.colors
        return self.pcl

    def visualize_pcd(self):
        if not self.isstarted:
            self.vis.add_geometry(self.pcl)
            origin = cph.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            self.vis.add_geometry(origin)
            self.isstarted = True
        else:
            self.vis.update_geometry(self.pcl)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()