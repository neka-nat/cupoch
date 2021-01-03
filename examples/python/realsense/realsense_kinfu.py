import sys
import pyrealsense2 as rs
import numpy as np
from enum import IntEnum

import cupoch as cph
cph.initialize_allocator(cph.PoolAllocation, 1000000000)


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = cph.camera.PinholeCameraIntrinsic(640, 480, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)
    return out


if __name__ == "__main__":

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3  # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    # print(depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    vis = cph.visualization.Visualizer()
    vis.create_window()

    mesh = cph.geometry.TriangleMesh()

    # Streaming loop
    frame_count = 0
    try:
        while True:

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            intrinsic = cph.camera.PinholeCameraIntrinsic(
                get_intrinsic_matrix(color_frame))

            if frame_count == 0:
                kinfu = cph.kinfu.Pipeline(intrinsic, cph.kinfu.KinfuParameters())

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = cph.geometry.Image(
                np.array(aligned_depth_frame.get_data()))
            color_temp = np.asarray(color_frame.get_data())
            color_image = cph.geometry.Image(color_temp)

            rgbd_image = cph.geometry.RGBDImage.create_from_color_and_depth(
                color_image.flip_horizontal(),
                depth_image.flip_horizontal(),
                depth_scale=1.0 / depth_scale,
                depth_trunc=clipping_distance_in_meters,
                convert_rgb_to_intensity=False)
            res = kinfu.process_frame(rgbd_image)
            if res:
                print(kinfu.cur_pose)
            temp = kinfu.extract_triangle_mesh()
            mesh.vertices = temp.vertices
            mesh.vertex_normals = temp.vertex_normals
            mesh.vertex_colors = temp.vertex_colors
            mesh.triangles = temp.triangles

            if frame_count == 0:
                vis.add_geometry(mesh)

            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()

            frame_count += 1

    finally:
        pipeline.stop()
    vis.destroy_window()