import rospy
import cupoch as cph
from sensor_msgs.msg import PointCloud2, PointField

if __name__ == "__main__":
    rospy.init_node("pointcloud2_msg", anonymous=True)
    rospy.loginfo("Start pointcloud2_msg node")
    pub = rospy.Publisher("pointcloud", PointCloud2, queue_size=1)

    pcd = cph.io.read_point_cloud("../../testdata/icp/cloud_bin_2.pcd")
    data, info = cph.io.create_to_pointcloud2_msg(pcd)
    msg = PointCloud2()
    msg.header.frame_id = "map"
    msg.width = info.width
    msg.height = info.height
    msg.fields = [PointField(name=f.name, offset=f.offset, datatype=f.datatype, count=f.count) for f in info.fields]
    msg.is_bigendian = info.is_bigendian
    msg.point_step = info.point_step
    msg.is_dense = info.is_dense
    msg.row_step = info.row_step
    msg.data = data
    print(msg)
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        pub.publish(msg)
        r.sleep()
