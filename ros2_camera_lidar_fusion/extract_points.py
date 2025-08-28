#!/usr/bin/env python3

import os
import cv2
import open3d as o3d
import numpy as np
from rclpy.node import Node
import rclpy

from ros2_camera_lidar_fusion.read_yaml import extract_configuration

class ImageCloudCorrespondenceNode(Node):
    def __init__(self):
        super().__init__('image_cloud_correspondence_node')

        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        self.data_dir = config_file['general']['data_folder']
        self.file = config_file['general']['correspondence_file']

        if not os.path.exists(self.data_dir):
            self.get_logger().warn(f"Data directory '{self.data_dir}' does not exist.")
            os.makedirs(self.data_dir)

        self.get_logger().info(f"Looking for .png and .pcd file pairs in '{self.data_dir}'")
        self.process_file_pairs()

    def get_file_pairs(self, directory):
        files = os.listdir(directory)
        pairs_dict = {}
        for f in files:
            full_path = os.path.join(directory, f)
            if not os.path.isfile(full_path):
                continue
            name, ext = os.path.splitext(f)

            if ext.lower() in [".png", ".jpg", ".jpeg", ".pcd"]:
                if name not in pairs_dict:
                    pairs_dict[name] = {}
                if ext.lower() == ".png":
                    pairs_dict[name]['png'] = full_path
                elif ext.lower() == ".pcd":
                    pairs_dict[name]['pcd'] = full_path

        file_pairs = []
        for prefix, d in pairs_dict.items():
            if 'png' in d and 'pcd' in d:
                file_pairs.append((prefix, d['png'], d['pcd']))

        file_pairs.sort()
        return file_pairs

    def pick_image_points(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            self.get_logger().error(f"Error loading image: {image_path}")
            return []

        points_2d = []
        window_name = "Select points on the image (press 'q' or ESC to finish)"

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points_2d.append((x, y))
                self.get_logger().info(f"Image: click at ({x}, {y})")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            display_img = img.copy()
            for pt in points_2d:
                cv2.circle(display_img, pt, 5, (0, 0, 255), -1)

            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(10)
            if key == 27 or key == ord('q'):
                break

        cv2.destroyWindow(window_name)
        return points_2d

    def pick_cloud_points(self, pcd_path):
        pcd = o3d.io.read_point_cloud(pcd_path)
        if pcd.is_empty():
            self.get_logger().error(f"Empty or invalid point cloud: {pcd_path}")
            return []

        self.get_logger().info("\n[Open3D Instructions]")
        self.get_logger().info("  - Shift + left click to select a point")
        self.get_logger().info("  - Use mouse drag + scroll for orbit/zoom")
        self.get_logger().info("  - Press 'F' to focus on geometry")
        self.get_logger().info("  - Press 'q' or ESC to close the window when finished\n")

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Select points on the cloud", width=1280, height=720)
        vis.add_geometry(pcd)

        render_opt = vis.get_render_option()
        render_opt.point_size = 3.5

        # Camera setup: allow free orbit/zoom
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)      # start closer
        ctr.set_front([0, 0, -1])
        ctr.set_lookat(pcd.get_center())
        ctr.set_up([0, -1, 0])

        vis.run()
        vis.destroy_window()

        picked_indices = vis.get_picked_points()
        np_points = np.asarray(pcd.points)
        picked_xyz = []
        for idx in picked_indices:
            xyz = np_points[idx]
            picked_xyz.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            self.get_logger().info(f"Cloud: index={idx}, coords=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")

        return picked_xyz


    def process_file_pairs(self):
        file_pairs = self.get_file_pairs(self.data_dir)
        if not file_pairs:
            self.get_logger().error(f"No .png / .pcd pairs found in '{self.data_dir}'")
            return

        self.get_logger().info("Found the following pairs:")
        for prefix, png_path, pcd_path in file_pairs:
            self.get_logger().info(f"  {prefix} -> {png_path}, {pcd_path}")

        for prefix, png_path, pcd_path in file_pairs:
            self.get_logger().info("\n========================================")
            self.get_logger().info(f"Processing pair: {prefix}")
            self.get_logger().info(f"Image: {png_path}")
            self.get_logger().info(f"Point Cloud: {pcd_path}")
            self.get_logger().info("========================================\n")

            image_points = self.pick_image_points(png_path)
            self.get_logger().info(f"Selected {len(image_points)} points in the image.\n")

            cloud_points = self.pick_cloud_points(pcd_path)
            self.get_logger().info(f"Selected {len(cloud_points)} points in the cloud.\n")

            out_txt = os.path.join(self.data_dir, self.file)
            with open(out_txt, 'a') as f:   # append mode
                f.write(f"# Pair: {prefix}\n")
                f.write("# u, v, x, y, z\n")
                min_len = min(len(image_points), len(cloud_points))
                for i in range(min_len):
                    (u, v) = image_points[i]
                    (x, y, z) = cloud_points[i]
                    f.write(f"{u},{v},{x},{y},{z}\n")
                f.write("\n")  # blank line between pairs

            self.get_logger().info(f"Saved {min_len} correspondences in: {out_txt}")
            self.get_logger().info("========================================")

        self.get_logger().info("\nProcessing complete! Correspondences saved for all pairs.")


def main(args=None):
    rclpy.init(args=args)
    node = ImageCloudCorrespondenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
