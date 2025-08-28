#!/usr/bin/env python3
import os
import cv2
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node

from ros2_camera_lidar_fusion.common.read_yaml import extract_configuration
from ros2_camera_lidar_fusion.common import utils

PIX_EPS   = 8        # px radius for auto-pick
SHORTLIST_K = 8      # how many nearest candidates to show
WIN_NAME = "correspondences"
FONT = cv2.FONT_HERSHEY_SIMPLEX

       
class CorrespondenceTool(Node):
    def __init__(self):
        super().__init__('correspondence_tool')

        cfg = extract_configuration()
        self.data_dir = cfg['general']['data_folder']
        self.out_txt  = os.path.join(self.data_dir, cfg['general']['correspondence_file'])
        intr_yaml     = os.path.join(cfg['general']['config_folder'],
                                     cfg['general']['camera_intrinsic_calibration'])
        self.K, self.D, (self.W, self.H) = utils.load_intrinsics(intr_yaml)

        guess_yaml = os.path.join(cfg['general']['config_folder'],
                                  cfg['general']['camera_extrinsic_calibration'])
        self.T_guess = None
        if os.path.isfile(guess_yaml):
            try:
                self.T_guess = utils.load_extrinsic(guess_yaml)
                self.get_logger().info("Using existing extrinsic as guess for assisted 3D picking.")
            except Exception as e:
                self.get_logger().warn(f"Failed to load extrinsic guess: {e}")

        self.pairs = self._scan_pairs(self.data_dir)
        if not self.pairs:
            self.get_logger().error(f"No .png/.pcd pairs found in '{self.data_dir}'")
            return

        # State for mouse callback
        self.click_queue = []          # list of (x,y) clicks
        self.current_img = None
        self.current_pcd = None
        self.uv_guess    = None        # Nx2 projected points from guess
        self.xyz_all     = None        # Nx3 original cloud points

        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 1280, 720)
        cv2.setMouseCallback(WIN_NAME, self._on_mouse)

        self.run()


    def _select_from_shortlist(self, base_img, click_uv):
        """
        Show top-K nearest projected points around click_uv on the image.
        User chooses with keys '1'..'K'. 'm' = open Open3D instead, 'c' = cancel.
        Returns 3D point (np.float32, shape(3,)) or None or the string 'open3d'.
        """
        if self.uv_guess is None or self.xyz_all is None:
            return 'open3d'  # no guess: fall back to Open3D

        uv = self.uv_guess
        d2 = np.sum((uv - np.float32(click_uv))[None, :]**2, axis=2).ravel() if uv.ndim==3 else np.sum((uv - np.float32(click_uv))**2, axis=1)
        order = np.argsort(d2)[:min(SHORTLIST_K, len(d2))]
        if order.size == 0:
            return 'open3d'

        vis = base_img.copy()
        for i, idx in enumerate(order, start=1):
            u, v = uv[idx]
            cv2.circle(vis, (int(u), int(v)), 6, (0, 255, 255), 2)
            cv2.putText(vis, str(i), (int(u)+8, int(v)-8), FONT, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, str(i), (int(u)+8, int(v)-8), FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(vis, "Pick 1..{}  |  m=Open3D  c=cancel".format(len(order)), (12, 32),
                    FONT, 0.7, (40, 230, 40), 2, cv2.LINE_AA)

        while True:
            cv2.imshow(WIN_NAME, vis)
            k = cv2.waitKey(0) & 0xFF
            if k in [ord(str(x)) for x in range(1, len(order)+1)]:
                sel = int(chr(k)) - 1
                return self.xyz_all[order[sel]].astype(np.float32)
            if k == ord('m'):
                return 'open3d'
            if k == ord('c') or k == 27:
                return None

    # ---------- IO helpers ----------
    def _scan_pairs(self, d):
        files = os.listdir(d); m = {}
        for f in files:
            n, e = os.path.splitext(f); p = os.path.join(d, f)
            if e.lower() == '.png': m.setdefault(n, {})['png'] = p
            if e.lower() == '.pcd': m.setdefault(n, {})['pcd'] = p
        pairs = []
        for k, v in sorted(m.items()):
            if 'png' in v and 'pcd' in v:
                pairs.append((k, v['png'], v['pcd']))
        return pairs

    # ---------- Mouse ----------
    def _on_mouse(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_queue.append((int(x), int(y)))

    # ---------- Assisted picking ----------
    def _prepare_guess_projection(self, pcd):
        if self.T_guess is None: 
            self.uv_guess = None; self.xyz_all = None; return
        xyz = np.asarray(pcd.points, dtype=np.float32)
        if xyz.shape[0] == 0:
            self.uv_guess = None; self.xyz_all = None; return
        pc = utils.transform_points(self.T_guess, xyz)
        uv = utils.project_points(pc, self.K, self.D)
        self.uv_guess = uv.astype(np.float32)
        self.xyz_all  = xyz

    def _autopick_3d(self, click_uv):
        if self.uv_guess is None or self.xyz_all is None:
            return None
        cu = np.array(click_uv, dtype=np.float32)
        d2 = np.sum((self.uv_guess - cu[None, :])**2, axis=1)
        idx = np.argmin(d2)
        if np.sqrt(d2[idx]) <= PIX_EPS:
            return self.xyz_all[idx].astype(np.float32)
        return None

    def _manual_pick_3d(self, pcd):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Pick one 3D point (Shift+LMB to add, Q to close)", width=1200, height=800)
        vis.add_geometry(pcd)

        # a smaller point size helps the giant sphere look less overwhelming
        ro = vis.get_render_option()
        ro.point_size = 3

        vis.run()
        inds = vis.get_picked_points()
        vis.destroy_window()

        # IMPORTANT: re-register OpenCV window + callback after Open3D
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 1280, 720)
        cv2.setMouseCallback(WIN_NAME, self._on_mouse)

        if not inds:
            return None
        p = np.asarray(pcd.points)[inds[0]].astype(np.float32)
        return p

    # ---------- Main loop ----------
    def run(self):
        for name, img_path, pcd_path in self.pairs:
            img = cv2.imread(img_path)
            pcd = o3d.io.read_point_cloud(pcd_path)
            if img is None or pcd.is_empty():
                self.get_logger().warn(f"Skip {name} (bad image/pcd).")
                continue

            self.current_img = img
            self.current_pcd = pcd
            self._prepare_guess_projection(pcd)

            uv_list = []
            xyz_list = []

            overlay = img.copy()
            msg = f"{name} | click=add, u=undo, s=save, q=quit"
            cv2.putText(overlay, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 230, 40), 1, cv2.LINE_AA)

            while True:
                show = overlay.copy()
                for (u, v) in uv_list:
                    cv2.circle(show, (int(u), int(v)), 4, (0, 0, 255), -1)

                cv2.imshow(WIN_NAME, show)
                k = cv2.waitKey(10) & 0xFF

                # handle pending clicks
                while self.click_queue:
                    (ux, vy) = self.click_queue.pop(0)

                    # 1) try strict autopick
                    p3 = self._autopick_3d((ux, vy))
                    if p3 is None:
                        # 2) shortlist UI on the image if guess exists
                        choice = self._select_from_shortlist(self.current_img, (ux, vy))
                        if isinstance(choice, str) and choice == 'open3d':
                            # 3) real manual pick in 3D
                            p3 = self._manual_pick_3d(self.current_pcd)
                        else:
                            p3 = choice

                    if p3 is None:
                        # user cancelled; continue without adding a pair
                        continue

                    uv_list.append((float(ux), float(vy)))
                    xyz_list.append((float(p3[0]), float(p3[1]), float(p3[2])))


                if k == ord('u') and uv_list:
                    uv_list.pop(); xyz_list.pop()
                elif k == ord('s'):
                    if len(uv_list) > 0:
                        utils.write_corresp_append(self.out_txt, name, uv_list, xyz_list)
                        self.get_logger().info(f"Saved {len(uv_list)} pairs for {name} → {self.out_txt}")
                    else:
                        self.get_logger().info(f"Find {len(uv_list)} pairs for {name}")

                    break
                elif k == ord('q'):
                    cv2.destroyWindow(WIN_NAME)
                    return

        cv2.destroyWindow(WIN_NAME)

def main(args=None):
    rclpy.init(args=args)
    node = CorrespondenceTool()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
