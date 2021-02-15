import rclpy
from rclpy.node import Node
import cupoch as cph
from sensor_msgs.msg import PointCloud2


class PointCloudSubscriber(Node):
    def __init__(self):
        super().__init__("pointcloud2_msg")
        self.subscription = self.create_subscription(
            PointCloud2,
            '/d435/camera/pointcloud',
            self.callback,
            10)
        self.subscription  # prevent unused variable warning
        self.pcd = cph.geometry.PointCloud()
        self.vis = cph.visualization.Visualizer()
        self.vis.create_window()
        self.initialize = False
 
    def callback(self, data):
        temp = cph.io.create_from_pointcloud2_msg(data.data.tobytes(),
                                                  cph.io.PointCloud2MsgInfo.default_dense(data.width,
                                                                                          data.height,
                                                                                          data.point_step))
        self.pcd.points = temp.points
        self.pcd.colors = temp.colors
        self.get_logger().info("%d" % len(self.pcd.points))

    def update_vis(self):
        if len(self.pcd.points) > 0:
            if not self.initialize:
                self.vis.add_geometry(self.pcd)
                self.initialize = True
            self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()


def main(args=None):
    rclpy.init(args=args)
    subscriber = PointCloudSubscriber()
    while rclpy.ok():
        rclpy.spin_once(subscriber)
        subscriber.update_vis()

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
