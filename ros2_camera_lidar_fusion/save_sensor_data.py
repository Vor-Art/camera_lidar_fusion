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

        # pubs
        projected_topic_pub = config_file['output']['projected_topic_pub']
        self.pub_image = self.create_publisher(Image, projected_topic_pub, 1)

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        self.get_logger().warn(f'Data will be saved at {self.storage_path}')

        self.image_sub = Subscriber(self, Image, self.image_topic)
        self.pointcloud_sub = Subscriber(self, PointCloud2, self.lidar_topic)

        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.pointcloud_sub],
            queue_size=10,
            slop=self.slop
        )
        self.ts.registerCallback(self.synchronize_data)

        # ---- Cached last synchronized pair + sync primitives ----
        self._cache_lock = threading.Lock()
        self._cached_image_msg = None
        self._cached_pointcloud_msg = None
        self._data_available = threading.Event()  # set when cache has a pair
        self._save_request = threading.Event()    # set when user pressed Enter

        # In auto mode (no keyboard), we save every synchronized pair.
        self.save_data_flag = not self.keyboard_listener_enabled
        if self.keyboard_listener_enabled:
            self.start_keyboard_listener()

    def start_keyboard_listener(self):
        """Starts a thread to listen for keyboard events."""
        def listen_for_space():
            while True:
                key = input("Press 'Enter' to save data (keyboard listener enabled): ")
                if key.strip() == '':
                    # Request save of last cached pair (if none yet, wait for it).
                    self._save_request.set()
                    saved_now = self._try_save_cached_pair()
                    if not saved_now:
                        self.get_logger().info("No cached pair yet — waiting for synchronized data...")
                        self._data_available.wait()  # wait until synchronize_data caches a pair
                        self._try_save_cached_pair()  # will save and reset cache
        thread = threading.Thread(target=listen_for_space, daemon=True)
        thread.start()

    def synchronize_data(self, image_msg, pointcloud_msg):
        """Handles synchronized messages and saves data if configured."""
        # Update cache with the latest synchronized pair.
        with self._cache_lock:
            self._cached_image_msg = image_msg
            self._cached_pointcloud_msg = pointcloud_msg
            self._data_available.set()

        self.pub_image.publish(image_msg)

        # If we are in auto-save mode, save every synchronized pair.
        if not self.keyboard_listener_enabled and self.save_data_flag:
            self._try_save_cached_pair()
            return

        # If user requested a save (pressed Enter earlier), fulfill it now.
        if self.keyboard_listener_enabled and self._save_request.is_set():
            self._try_save_cached_pair()

    def _try_save_cached_pair(self) -> bool:
        """Save the currently cached pair (if any) and reset the cache. Returns True if saved."""
        with self._cache_lock:
            image_msg = self._cached_image_msg
            pointcloud_msg = self._cached_pointcloud_msg

            if image_msg is None or pointcloud_msg is None:
                return False

            # Generate name and enforce storage limit.
            file_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            try:
                # Count pairs as .png/.pcd; keep user's simple heuristic (files/2).
                total_files = len(os.listdir(self.storage_path)) / 2.0
            except Exception as e:
                self.get_logger().error(f'Failed to count files: {e}')
                total_files = 0

            if total_files >= self.max_file_saved:
                self.get_logger().info(f'To many data total_files={total_files}')
                # Do not clear save request so user can retry later if desired.
                return False

            self.get_logger().info(f'Saving cached synchronized data as {file_name}')
            # Perform save under lock to keep pair consistent.
            try:
                self.save_data(image_msg, pointcloud_msg, file_name)
            except Exception as e:
                self.get_logger().error(f'Failed to save data: {e}')
                return False

            # Reset cache after successful save so next Enter won't re-save the same pair.
            self._cached_image_msg = None
            self._cached_pointcloud_msg = None
            self._data_available.clear()
            self._save_request.clear()
            return True

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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
