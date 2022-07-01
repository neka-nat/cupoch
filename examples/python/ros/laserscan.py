import numpy as np
import cupoch as cph
import rosbag
import tf.transformations as tftrans

bagfile = "../../testdata/Mapping1.bag"
bag = rosbag.Bag(bagfile)
initialize = False
scan = None
trans = np.identity(4)
vis = cph.visualization.Visualizer()
vis.create_window()
pcd = cph.geometry.PointCloud()
frame = 0
n_buf = 50
for topic, msg, t in bag.read_messages():
    if topic == "/base_pose_ground_truth":
        trans = tftrans.quaternion_matrix(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        )
        trans[0:3, 3] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        frame += 1

    if topic == "/base_scan":
        if not initialize:
            scan = cph.geometry.LaserScanBuffer(len(msg.ranges), n_buf, msg.angle_min, msg.angle_max)
            initialize = True
        if not scan is None:
            scan.add_ranges(cph.utility.FloatVector(msg.ranges), trans, cph.utility.FloatVector(msg.intensities))

    if initialize and frame % n_buf == 0:
        temp = cph.geometry.PointCloud.create_from_laserscanbuffer(scan, 0.0, 10.0)
        pcd += temp
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

vis.destroy_window()
