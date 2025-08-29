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
from ros2_camera_lidar_fusion.common import utils, densify_utils
from ros2_camera_lidar_fusion.common.profiller import Profiler


class LidarCameraProjectionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_projection_node')

        cfg = extract_configuration()
        cfg_folder = cfg['general']['config_folder']
        extrinsic_yaml = os.path.join(cfg_folder, cfg['general']['camera_extrinsic_calibration'])
        intr_yaml      = os.path.join(cfg_folder, cfg['general']['camera_intrinsic_calibration'])

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

        # densify params
        dp = cfg['densify_parameters']
        self.voxel_size_m     = float(dp['voxel_size_m'])
        # densify params (depth_uniform)
        dp_u = dp['depth_uniform']
        self.en_depth_uniform = bool(dp_u['enable'])
        self.grid_step_px     = int(dp_u['grid_step_px'])
        self.fill_iters       = int(dp_u['fill_iters'])
        self.max_interp_dist  = float(dp_u['max_interp_dist'])
        # densify params (Velodyne geometry)
        dp_v = dp['velodyne_rings']
        self.en_vlp_densify   = bool(dp_v['enable'])
        self.num_rings        = int(dp_v['num_rings'])
        self.az_bins          = int(dp_v['az_bins'])
        self.interp_per_gap   = int(dp_v['interp_per_gap'])
        self.max_range_jump   = float(dp_v['max_range_jump'])
        self.max_gap_m        = None if dp_v['max_gap_m'] is None else float(dp_v['max_gap_m'])

        # profiling
        prof_cfg = cfg.get('profiling', {}) if isinstance(cfg, dict) else {}
        self.prof = Profiler(
            log_every=int(prof_cfg.get('log_every', 10)),
            window_sec=float(prof_cfg.get('window_sec', 5.0)),
            logger=self.get_logger()
        )

        self.get_logger().info(f"Loaded extrinsic:\n{self.T_lidar_to_cam}")
        self.get_logger().info(f"Loaded intrinsics:\n{self.camera_matrix}")

    # -------------- main callback --------------
    def sync_callback(self, image_msg: Image, lidar_msg: PointCloud2):
        t0_cb = self.prof.t(); self.prof.add_period(t0_cb)

        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        xyz_lidar = utils.pointcloud2_to_xyz_array_fast(lidar_msg)

        # --- Pre-filter in image frame (get only visible points) ---
        t0 = self.prof.t()
        xyz_work, dbg_img = densify_utils.filter_points_on_image(
            xyz_lidar, cv_image, self.T_lidar_to_cam, self.camera_matrix, self.dist_coeffs
        )
        self.prof.add("filtering", self.prof.t() - t0)

        # --- Publish debug image ---
        self.publish_debug_img(dbg_img, image_msg.header)


        # --- Densify 1: Velodyne rings (LiDAR frame) ---
        if self.en_vlp_densify:
            t0 = self.prof.t()
            xyz_work = densify_utils.densify_velodyne_rings(
                xyz_work,
                num_rings=self.num_rings,
                az_bins=self.az_bins,
                interp_per_gap=self.interp_per_gap,
                max_range_jump=self.max_range_jump,
                max_gap_m=self.max_gap_m,
            )
            self.prof.add("vlp_densify", self.prof.t() - t0)

        # --- Recompute params ---
        mask, u, v, z = densify_utils.extract_lidar_image_params(
            xyz_work, cv_image, self.T_lidar_to_cam, self.camera_matrix, self.dist_coeffs
        )
        xyz_work = xyz_work[mask]
        px_work = np.column_stack((u, v))

        # --- Densify 2: Depth-uniform (image plane) ---
        if self.en_depth_uniform:
            t0 = self.prof.t()
            xyz_cam_dense, px_work = densify_utils.densify_depth_uniform(
                u, v, z, img=cv_image,
                camera_matrix=self.camera_matrix,
                grid_step_px=self.grid_step_px,
                fill_iters=self.fill_iters,
                max_interp_dist=self.max_interp_dist
            )
            xyz_work = utils.transform_points(self.T_cam_to_lidar, xyz_cam_dense)
            self.prof.add("depth_densify", self.prof.t() - t0)

        # --- Voxel colorize ---
        t0 = self.prof.t()
        xyz_v, rgb_v = utils.voxel_downsample_with_color(xyz_work, px_work, cv_image, self.voxel_size_m)
        self.prof.add("voxelize", self.prof.t() - t0)

        # --- Publish cloud ---
        t0 = self.prof.t()
        self.publish_colored_cloud(xyz_v, rgb_v, lidar_msg.header)
        self.prof.add("publish_cloud", self.prof.t() - t0)

        self.prof.add("total", self.prof.t() - t0_cb)
        self.prof.maybe_log()

    def publish_debug_img(self, image, header):
        if self.flip_method in [0, 1, -1]:
            cv_image = cv2.flip(image, self.flip_method)
        out_img = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        out_img.header = header
        self.pub_image.publish(out_img)

    def publish_colored_cloud(self, xyz_points, rgb_float, msg_header):
        if xyz_points.shape[0] == 0:
            return
        pts = np.column_stack((xyz_points.astype(np.float32), rgb_float.astype(np.float32)))
        fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        hdr = Header()
        hdr.stamp = msg_header.stamp
        hdr.frame_id = msg_header.frame_id or "lidar"
        cloud_msg = pc2.create_cloud(hdr, fields, pts.tolist())
        self.pub_cloud.publish(cloud_msg)


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
