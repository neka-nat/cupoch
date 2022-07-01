import rospy
import cupoch as cph
from sensor_msgs.msg import Image

if __name__ == "__main__":
    rospy.init_node("image_msg", anonymous=True)
    rospy.loginfo("Start image_msg node")
    img = cph.geometry.Image()
    vis = cph.visualization.Visualizer()
    vis.create_window()

    def callback(data):
        global img
        img = cph.io.create_from_image_msg(data.data, cph.io.ImageMsgInfo.default_cv_color(data.width, data.height))
        rospy.loginfo("%d, %d" % (img.width, img.height))

    rospy.Subscriber("/image_publisher/image_raw", Image, callback)
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        vis.clear_geometries()
        vis.add_geometry(img)
        vis.poll_events()
        vis.update_renderer()
        r.sleep()
