import sys
import pyrealsense2 as rs
import numpy as np
from enum import IntEnum
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from datetime import datetime

import cupoch as x3d
x3d.initialize_allocator(x3d.PoolAllocation, 1000000000)


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = x3d.camera.PinholeCameraIntrinsic(640, 480, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)
    return out


if __name__ == "__main__":

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 4)

    # Streaming loop
    prev_rgbd_image = None
    option = x3d.odometry.OdometryOption()
    option.inv_sigma_mat_diag = 0.5 * np.ones(6)
    cur_trans = np.identity(4)
    twist = np.zeros(6)
    path = []
    path.append(cur_trans[:3, 3].tolist())
    line = ax1.plot(*list(zip(*path)), 'r-')[0]
    pos = ax1.plot(*list(zip(*path)), 'yo')[0]
    ax1.invert_yaxis()
    depth_im = ax2.imshow(np.zeros((480, 640), dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
    color_im = ax3.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    try:
        def update_odom(frame):
            global prev_rgbd_image, cur_trans, twist

            dt = datetime.now()

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            intrinsic = x3d.camera.PinholeCameraIntrinsic(
                get_intrinsic_matrix(color_frame))

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                return

            depth_temp = np.array(aligned_depth_frame.get_data())
            depth_image = x3d.geometry.Image(depth_temp)
            color_temp = np.asarray(color_frame.get_data())
            color_image = x3d.geometry.Image(color_temp)

            rgbd_image = x3d.geometry.RGBDImage.create_from_color_and_depth(color_image,
                                                                            depth_image)

            if not prev_rgbd_image is None:
                res, odo_trans, twist, _ = x3d.odometry.compute_weighted_rgbd_odometry(
                                prev_rgbd_image, rgbd_image, intrinsic,
                                np.identity(4), twist,
                                x3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
                if res:
                    cur_trans = np.matmul(cur_trans, odo_trans)

            prev_rgbd_image = rgbd_image
            process_time = datetime.now() - dt
            print("FPS: " + str(1 / process_time.total_seconds()))
            print(cur_trans)
            path.append(cur_trans[:3, 3])
            data = list(zip(*path))
            line.set_data(np.array(data[0]), np.array(data[1]))
            line.set_3d_properties(data[2])
            pos.set_data(np.array([cur_trans[0, 3]]), np.array([cur_trans[1, 3]]))
            pos.set_3d_properties(cur_trans[2, 3])
            depth_offset = depth_temp.min()
            depth_scale = depth_temp.max() - depth_offset
            depth_temp = np.clip((depth_temp - depth_offset) / depth_scale, 0.0, 1.0)
            depth_im.set_array((255.0 * depth_temp).astype(np.uint8))
            color_im.set_array(color_temp)

        anim = animation.FuncAnimation(fig, update_odom, interval=10)
        plt.show()

    finally:
        pipeline.stop()