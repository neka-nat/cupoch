import rospy
import cupoch as cph
from sensor_msgs.msg import PointCloud2

if __name__ == "__main__":
    rospy.init_node('pointcloud2_msg', anonymous=True)
    rospy.loginfo("Start pointcloud2_msg node")
    pcd = cph.geometry.PointCloud()
    
    def callback(data):
        global pcd
        pcd = cph.io.create_from_pointcloud2_msg(data.data,
                                                 cph.io.PointCloud2MsgInfo.default(data.width,
                                                                                   data.point_step))
        rospy.loginfo("%d" % len(pcd.points))

    rospy.Subscriber("/camera/depth/color/points", PointCloud2, callback)
    rospy.spin()