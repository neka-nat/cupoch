import numpy as np
import cupoch as cph
import rosbag
from sensor_msgs.msg import LaserScan
import transformations as tftrans


bagfile = "../../testdata/Mapping1.bag"
bag = rosbag.Bag(bagfile)
initialize = False
scan = None
trans = np.identity(4)
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
        if scan is not None:
            scan.add_host_ranges(
                cph.utility.HostFloatVector(msg.ranges),
                trans,
                cph.utility.HostFloatVector(msg.intensities),
            )

            new_msg = LaserScan()
            new_msg.angle_min = scan.min_angle
            new_msg.angle_max = scan.max_angle
            new_msg.angle_increment = (scan.max_angle - scan.min_angle) / scan.num_steps
            new_msg.range_min = 0.0
            new_msg.range_max = 10000.0
            new_msg.ranges = scan.pop_host_one_scan()
            print("------ Original laserscan ------ \n", msg)
            print("------ Converted laserscan ------\n", new_msg)
