#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
import yaml
from datetime import datetime

from ros2_camera_lidar_fusion.common.read_yaml import extract_configuration


class CameraInfoSaver(Node):
    def __init__(self):
        super().__init__('camera_info_saver')

        cfg = extract_configuration()
        if cfg is None:
            self.get_logger().error("Failed to load configuration.")
            rclpy.shutdown()
            return

        self.info_topic   = cfg['camera']['info_topic']
        self.output_dir   = cfg['general']['config_folder']
        self.output_file  = cfg['general']['camera_intrinsic_calibration']
        os.makedirs(self.output_dir, exist_ok=True)
        self.saved_once = False

        self.create_subscription(CameraInfo, self.info_topic, self.camera_info_cb, 10)
        self.get_logger().info(f"Waiting for CameraInfo on '{self.info_topic}' ...")

    def camera_info_cb(self, msg: CameraInfo):
        if self.saved_once:
            return
        self.saved_once = True

        # force conversion to plain python types
        K = [float(x) for x in msg.k]       # 9 values row-major
        D = [float(x) for x in msg.d] if msg.d else []
        width, height = int(msg.width), int(msg.height)

        data = {
            'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'camera_matrix': {
                'rows': 3, 'columns': 3, 'data': K
            },
            'distortion_coefficients': {
                'rows': 1, 'columns': len(D), 'data': D
            },
            'image_size': {
                'width': width, 'height': height
            }
        }

        out_path = os.path.join(self.output_dir, self.output_file)
        try:
            with open(out_path, 'w') as f:
                yaml.safe_dump(data, f, sort_keys=False)
            self.get_logger().info(f"Intrinsics saved to: {out_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save intrinsics: {e}")

        self.get_logger().info("Done. Shutting down.")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
