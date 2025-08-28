#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header

from ros2_camera_lidar_fusion.common.read_yaml import extract_configuration
from ros2_camera_lidar_fusion.common import utils

class LidarCameraProjectionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_projection_node')

        cfg = extract_configuration()
        cfg_folder = cfg['general']['config_folder']

        extrinsic_yaml = os.path.join(cfg_folder, cfg['general']['camera_extrinsic_calibration'])
        intr_yaml     = os.path.join(cfg_folder, cfg['general']['camera_intrinsic_calibration'])

        self.T_lidar_to_cam = utils.load_extrinsic(extrinsic_yaml)
        self.T_cam_to_lidar = utils.invert_h(self.T_lidar_to_cam)
        self.camera_matrix, self.dist_coeffs, _ = utils.load_intrinsics(intr_yaml)

        self.flip_method = cfg['output']['flip_method']

        # subs + sync
        image_topic = cfg['input']['image_topic']
        lidar_topic = cfg['input']['lidar_topic']
        self.image_sub = Subscriber(self, Image, image_topic)
        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic)
        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.lidar_sub], queue_size=5, slop=0.07)
        self.ts.registerCallback(self.sync_callback)

        # pubs
        projected_topic_pub = cfg['output']['projected_topic_pub']
        lidar_in_camera_pub = cfg['output']['lidar_in_camera_pub']
        self.pub_image = self.create_publisher(Image, projected_topic_pub, 1)
        self.pub_cloud = self.create_publisher(PointCloud2, lidar_in_camera_pub, 1)

        self.bridge = CvBridge()
        self.skip_rate = 1

        # densify parameters
        dp = cfg['densify_parameters']
        self.grid_step_px   = int(dp['grid_step_px'])   # uniform spacing on image plane
        self.fill_iters     = int(dp['fill_iters'])     # hole-fill iterations
        self.voxel_size_m   = float(dp['voxel_size_m']) # voxel size in LiDAR frame (m)
        self.max_interp_dist   = float(dp['max_interp_dist']) # distance to interpolate (m)

        self.get_logger().info(f"Loaded extrinsic:\n{self.T_lidar_to_cam}")
        self.get_logger().info(f"Loaded intrinsics:\n{self.camera_matrix}")

    def sync_callback(self, image_msg: Image, lidar_msg: PointCloud2):
        if self.camera_matrix is None:
            return

        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # LiDAR points
        xyz_lidar = utils.pointcloud2_to_xyz_array_fast(lidar_msg, skip_rate=self.skip_rate)
        if xyz_lidar.shape[0] == 0:
            return

        # LiDAR → Camera
        xyz_cam = utils.transform_points(self.T_lidar_to_cam, xyz_lidar)

        # keep only points in front
        mask_in_front = xyz_cam[:, 2] > 0.0
        xyz_cam = xyz_cam[mask_in_front]
        if xyz_cam.shape[0] == 0:
            return

        # Project to 2D
        rvec = np.zeros((3,1)); tvec = np.zeros((3,1))
        image_points, _ = cv2.projectPoints(xyz_cam, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        image_points = image_points.reshape(-1, 2)

        # Draw on image
        cv_image_draw = cv_image.copy()
        h, w = cv_image_draw.shape[:2]
        u = np.round(image_points[:, 0]).astype(np.int32)
        v = np.round(image_points[:, 1]).astype(np.int32)
        mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(mask):
            return

        xyz_cam = xyz_cam[mask]; u = u[mask]; v = v[mask]; image_points = image_points[mask]
        for uu, vv in zip(u, v):
            cv2.circle(cv_image_draw, (int(uu), int(vv)), 2, (0, 255, 0), -1)

        if self.flip_method in [0, 1, -1]:
            cv_image_draw = cv2.flip(cv_image_draw, self.flip_method)

        out_img = self.bridge.cv2_to_imgmsg(cv_image_draw, encoding='bgr8')
        out_img.header = image_msg.header
        self.pub_image.publish(out_img)

        # Densify on image plane, back-project, voxel-colorize, publish
        xyz_dense, px_dense = self.densify_depth_uniform(u, v, xyz_cam[:, 2], cv_image, self.max_interp_dist)
        xyz_dense_lidar = utils.transform_points(self.T_cam_to_lidar, xyz_dense)

        hdr_dense = Header()
        hdr_dense.stamp = image_msg.header.stamp
        hdr_dense.frame_id = lidar_msg.header.frame_id or "lidar"

        xyz_v, rgb_v = utils.voxel_downsample_with_color(xyz_dense_lidar, px_dense, cv_image, self.voxel_size_m)
        self.publish_colored_cloud(xyz_v, rgb_v, hdr_dense)

    def publish_colored_cloud(self, xyz_points, rgb_float, header):
        if xyz_points.shape[0] == 0:
            return
        pts = np.column_stack((xyz_points.astype(np.float32), rgb_float.astype(np.float32)))
        fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg = pc2.create_cloud(header, fields, pts.tolist())
        self.pub_cloud.publish(cloud_msg)

    def densify_depth_uniform(self, u, v, z, img, max_interp_dist=0.2):
        h, w = img.shape[:2]
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        # --- z-buffer at pixel centers (fast) ---
        depth = np.full((h * w,), np.inf, np.float32)
        idx = (v.astype(np.int64) * w + u.astype(np.int64))
        np.minimum.at(depth, idx, z.astype(np.float32))
        depth = depth.reshape(h, w)

        # --- iterative neighbor-averaging hole fill ---
        for _ in range(max(1, self.fill_iters)):
            valid = np.isfinite(depth)
            num = cv2.blur(np.where(valid, depth, 0.0), (3, 3), borderType=cv2.BORDER_REPLICATE)
            den = cv2.blur(valid.astype(np.float32), (3, 3), borderType=cv2.BORDER_REPLICATE)

            # candidate fill positions
            fill_mask = ~valid & (den > 0)
            if not np.any(fill_mask):
                break

            # average depth from neighbors
            avg_depth = num / np.maximum(den, 1e-6)

            # distance check: only fill if neighbors are close in depth
            # compute local min/max depth in 3x3 window
            min_d = cv2.erode(np.where(valid, depth, np.inf), np.ones((3, 3), np.uint8))
            max_d = cv2.dilate(np.where(valid, depth, -np.inf), np.ones((3, 3), np.uint8))
            ok = (max_d - min_d) < max_interp_dist

            # only accept pixels where disparity is small
            final_mask = fill_mask & ok
            depth[final_mask] = avg_depth[final_mask]

        # --- uniform sampling on a grid ---
        xs = np.arange(0, w, max(1, self.grid_step_px), dtype=np.int32)
        ys = np.arange(0, h, max(1, self.grid_step_px), dtype=np.int32)
        gx, gy = np.meshgrid(xs, ys)
        gx = gx.ravel(); gy = gy.ravel()

        good = np.isfinite(depth[gy, gx])
        if not np.any(good):
            return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int32)

        gx = gx[good]; gy = gy[good]
        d  = depth[gy, gx].astype(np.float64)

        X = (gx.astype(np.float64) - cx) / fx * d
        Y = (gy.astype(np.float64) - cy) / fy * d
        Z = d
        xyz_cam = np.column_stack((X, Y, Z)).astype(np.float32)
        px = np.column_stack((gx, gy)).astype(np.int32)
        return xyz_cam, px

def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
