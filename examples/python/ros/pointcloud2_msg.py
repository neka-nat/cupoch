import rospy
import cupoch as cph
from sensor_msgs.msg import PointCloud2

if __name__ == "__main__":
    rospy.init_node("pointcloud2_msg", anonymous=True)
    rospy.loginfo("Start pointcloud2_msg node")
    pcd = cph.geometry.PointCloud()
    vis = cph.visualization.Visualizer()
    vis.create_window()

    def callback(data):
        global pcd
        temp = cph.io.create_from_pointcloud2_msg(
            data.data, cph.io.PointCloud2MsgInfo.default(data.width, data.point_step)
        )
        pcd.points = temp.points
        pcd.colors = temp.colors
        rospy.loginfo("%d" % len(pcd.points))

    rospy.Subscriber("/camera/depth/color/points", PointCloud2, callback)
    r = rospy.Rate(10)
    initialize = False
    while not rospy.is_shutdown():
        if len(pcd.points) > 0:
            if not initialize:
                vis.add_geometry(pcd)
                initialize = True
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
        r.sleep()
