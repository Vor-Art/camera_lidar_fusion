#!/usr/bin/env python3

import rclpy, os, cv2, datetime
import numpy as np
from cv_bridge import CvBridge
import open3d as o3d
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import threading

from ros2_camera_lidar_fusion.common.read_yaml import extract_configuration

class SaveData(Node):
    def __init__(self):
        super().__init__('save_data_node')
        self.get_logger().info('Save data node has been started')

        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        self.max_file_saved = config_file['general']['max_file_saved']
        self.storage_path = config_file['general']['data_folder']
        self.image_topic = config_file['input']['image_topic']
        self.lidar_topic = config_file['input']['lidar_topic']
        self.keyboard_listener_enabled = config_file['general']['keyboard_listener']
        self.slop = config_file['general']['slop']


        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        self.get_logger().warn(f'Data will be saved at {self.storage_path}')

        self.image_sub = Subscriber(
            self,
            Image,
            self.image_topic
        )
        self.pointcloud_sub = Subscriber(
            self,
            PointCloud2,
            self.lidar_topic
        )

        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.pointcloud_sub],
            queue_size=10,
            slop=self.slop
        )
        self.ts.registerCallback(self.synchronize_data)

        self.save_data_flag = not self.keyboard_listener_enabled
        if self.keyboard_listener_enabled:
            self.start_keyboard_listener()

    def start_keyboard_listener(self):
        """Starts a thread to listen for keyboard events."""
        def listen_for_space():
            while True:
                key = input("Press 'Enter' to save data (keyboard listener enabled): ")
                if key.strip() == '':
                    self.save_data_flag = True
                    self.get_logger().info('Space key pressed, ready to save data')
        thread = threading.Thread(target=listen_for_space, daemon=True)
        thread.start()

    def synchronize_data(self, image_msg, pointcloud_msg):
        """Handles synchronized messages and saves data if the flag is set."""
        if self.save_data_flag:

            file_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            total_files = len(os.listdir(self.storage_path))/2
            if total_files < self.max_file_saved:
                self.get_logger().info(f'Synchronizing data at {file_name}')
                self.save_data(image_msg, pointcloud_msg, file_name)
            else:
                self.get_logger().info(f'To many data {total_files=}')
                
            if self.keyboard_listener_enabled:
                self.save_data_flag = False
    def pointcloud2_to_open3d(self, pointcloud_msg):
        """Converts a PointCloud2 message to an Open3D point cloud."""
        points = []
        for p in point_cloud2.read_points(pointcloud_msg, skip_nans=True):
            points.append([p[0], p[1], p[2]])
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float32))
        return pointcloud

    def save_data(self, image_msg, pointcloud_msg, file_name):
        """Saves image and point cloud data to the storage path."""
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        pointcloud = self.pointcloud2_to_open3d(pointcloud_msg)
        o3d.io.write_point_cloud(f'{self.storage_path}/{file_name}.pcd', pointcloud)
        cv2.imwrite(f'{self.storage_path}/{file_name}.png', image)
        self.get_logger().info(f'Data has been saved at {self.storage_path}/{file_name}.png')


def main(args=None):
    rclpy.init(args=args)
    node = SaveData()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
