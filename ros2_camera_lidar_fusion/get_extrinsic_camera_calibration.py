#!/usr/bin/env python3

import os
import yaml
import numpy as np
import cv2
from rclpy.node import Node
import rclpy

from ros2_camera_lidar_fusion.read_yaml import extract_configuration

class CameraLidarExtrinsicNode(Node):
    def __init__(self):
        super().__init__('camera_lidar_extrinsic_node')
        
        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        self.corr_file = config_file['general']['correspondence_file']
        self.corr_file = f'/ros2_ws/src/ros2_camera_lidar_fusion/data/{self.corr_file}'
        self.camera_yaml = config_file['general']['camera_intrinsic_calibration']
        self.camera_yaml = f'/ros2_ws/src/ros2_camera_lidar_fusion/config/{self.camera_yaml}'
        self.output_dir = config_file['general']['config_folder']
        self.file = config_file['general']['camera_extrinsic_calibration']

        self.get_logger().info('Starting extrinsic calibration...')
        self.solve_extrinsic_with_pnp()

    def load_camera_calibration(self, yaml_path: str):
        """Loads camera calibration parameters from a YAML file."""
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"Calibration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        mat_data = config['camera_matrix']['data']
        camera_matrix = np.array(mat_data, dtype=np.float64)
        dist_data = config['distortion_coefficients']['data']
        dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))

        return camera_matrix, dist_coeffs

    def solve_extrinsic_with_pnp(self):
        """Solves for extrinsic parameters using 2D-3D correspondences and camera calibration."""
        camera_matrix, dist_coeffs = self.load_camera_calibration(self.camera_yaml)
        self.get_logger().info(f"Camera matrix:\n{camera_matrix}")
        self.get_logger().info(f"Distortion coefficients: {dist_coeffs}")

        if not os.path.isfile(self.corr_file):
            raise FileNotFoundError(f"Correspondence file not found: {self.corr_file}")

        pts_2d = []
        pts_3d = []
        with open(self.corr_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                splitted = line.split(',')
                if len(splitted) != 5:
                    continue
                u, v, X, Y, Z = [float(val) for val in splitted]
                pts_2d.append([u, v])
                pts_3d.append([X, Y, Z])

        pts_2d = np.array(pts_2d, dtype=np.float64)
        pts_3d = np.array(pts_3d, dtype=np.float64)

        num_points = len(pts_2d)
        self.get_logger().info(f"Loaded {num_points} correspondences from {self.corr_file}")

        if num_points < 4:
            raise ValueError("At least 4 correspondences are required for solvePnP")

        success, rvec, tvec = cv2.solvePnP(
            pts_3d,
            pts_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            raise RuntimeError("solvePnP failed to find a solution.")

        self.get_logger().info("solvePnP succeeded.")
        self.get_logger().info(f"rvec: {rvec.ravel()}")
        self.get_logger().info(f"tvec: {tvec.ravel()}")

        R, _ = cv2.Rodrigues(rvec)

        T_lidar_to_cam = np.eye(4, dtype=np.float64)
        T_lidar_to_cam[0:3, 0:3] = R
        T_lidar_to_cam[0:3, 3] = tvec[:, 0]

        self.get_logger().info(f"Transformation matrix (LiDAR -> Camera):\n{T_lidar_to_cam}")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Save as YAML (human-readable, aligned YAML-like)
        out_yaml = os.path.join(self.output_dir, self.file)
        
        R = T_lidar_to_cam[0:3, 0:3]
        t = T_lidar_to_cam[0:3, 3]
        R_flat = R.flatten()
        t_list = t

        with open(out_yaml, 'w') as f:
            f.write("\n")
            f.write("# (LiDAR -> Camera)\n")

            # translation
            t_str = ", ".join(f"{val: .8f}" for val in t_list)
            f.write(f"translation: [{t_str}]\n")

            # rotation (flattened, 3 per row)
            f.write("rotation:    [")
            for i, val in enumerate(R_flat):
                sep = "," if i < len(R_flat)-1 else ""
                f.write(f"{val: .8f}{sep} ")
                if (i+1) % 3 == 0 and i < len(R_flat)-1:
                    f.write("\n              ")
            f.write("]\n")

            # transformation matrix
            f.write("transformation_matrix:\n")
            for row in T_lidar_to_cam:
                row_str = ", ".join(f"{val: .8f}" for val in row)
                f.write(f"  - [{row_str}]\n")


        

        self.get_logger().info(f"Extrinsic matrix saved to: {out_yaml}")


def main(args=None):
    rclpy.init(args=args)
    node = CameraLidarExtrinsicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
