#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node

from ros2_camera_lidar_fusion.common.read_yaml import extract_configuration
from ros2_camera_lidar_fusion.common import utils
from ros2_camera_lidar_fusion import get_extrinsic_camera_calibration as calib
PIX_EPS = 10
IMG_WIN = "image_correspondences"

# ---------- Open3D GUI wrapper ----------
class PCDGui:
    def __init__(self, title="pcd_correspondences", width=1200, height=800):
        gui = o3d.visualization.gui
        rendering = o3d.visualization.rendering

        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window(title, width, height)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        self._pt_size = 3.0        # default point size

        # point cloud material (point size lives here)
        self.pcd_mat = rendering.MaterialRecord()
        self.pcd_mat.shader = "defaultUnlit"
        self.pcd_mat.point_size = float(self._pt_size)

        # marker material (solid color, independent of point size)
        self.marker_mat = rendering.MaterialRecord()
        self.marker_mat.shader = "unlitSolidColor"
        self.marker_mat.base_color = (1.0, 0.0, 0.0, 1.0)

        self._click_queue = []     # (x, y) pixels in widget coords
        self._key_queue = []       # 'u','s','q'
        self._markers = []         # [(name_sphere, label_handle|None)]
        self._pcd_name = "pcd"
        self.T_guess = None

        self._last_down_t = 0.0
        self._dbl_thr_s = 0.35

        # # --- compat shim for return values ---
        try:
            ECR = gui.EventCallbackResult
            RET_CONSUMED, RET_HANDLED, RET_IGNORED = ECR.CONSUMED, ECR.HANDLED, ECR.IGNORED
        except AttributeError:  # older wheels
            RET_CONSUMED, RET_HANDLED, RET_IGNORED = 2, 1, 0

        # Mouse → collect LMB clicks
        def on_mouse(e):
            if (e.type == gui.MouseEvent.Type.BUTTON_DOWN and
                e.is_button_down(gui.MouseButton.LEFT)):
        
                now = time.monotonic()
                if (now - self._last_down_t) < self._dbl_thr_s:
                    self._last_down_t = now
                    return RET_CONSUMED

                self._click_queue.append((int(e.x), int(e.y)))
                self._last_down_t = now
                return RET_HANDLED
            return RET_IGNORED

        # Keys → mirror image window controls
        def on_key(e):
            if e.type == gui.KeyEvent.Type.DOWN:
                # printable ASCII
                if 0 <= e.key < 256:
                    ch = chr(e.key).lower()
                    self._key_queue.append(ch)
                    return RET_HANDLED
            return RET_IGNORED

        # register callbacks
        self.scene.set_on_mouse(on_mouse)
        self.window.set_on_key(on_key)

        # keep the SceneWidget filling the window
        def on_layout(ctx):
            r = self.window.content_rect
            self.scene.frame = r
        self.window.set_on_layout(on_layout)

    def set_point_size(self, px: float):
        self._pt_size = max(1.0, float(px))  # clamp if you want
        self.pcd_mat.point_size = self._pt_size
        try:
            self.scene.scene.modify_geometry_material(self._pcd_name, self.pcd_mat)
        except Exception:
            pass

    def clear(self):
        # remove old markers
        for name, lbl in self._markers:
            try:
                self.scene.scene.remove_geometry(name)
            except Exception:
                pass
            try:
                if lbl is not None:
                    self.scene.remove_3d_label(lbl)
            except Exception:
                pass
        self._markers.clear()

    def set_pointcloud(self, pcd: o3d.geometry.PointCloud):
        self.clear()
        try:
            self.scene.scene.remove_geometry(self._pcd_name)
        except Exception:
            pass
        self.scene.scene.add_geometry(self._pcd_name, pcd, self.pcd_mat)
        bounds = pcd.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60.0, bounds, bounds.get_center())
        self.scene.scene.show_axes(False)

    def poll(self):
        self.app.run_one_tick()

    def pop_click(self):
        return self._click_queue.pop(0) if self._click_queue else None

    def pop_key(self):
        return self._key_queue.pop(0) if self._key_queue else None

    def add_marker(self, position: np.ndarray, idx: int):
        # small red sphere + 3D label with index
        sph = o3d.geometry.TriangleMesh.create_sphere(0.03)
        sph.compute_vertex_normals()
        sph.paint_uniform_color([1.0, 0.0, 0.0])
        sph.translate(position.astype(float))

        name = f"marker_{idx}"
        self.scene.scene.add_geometry(name, sph, self.marker_mat)
        lbl = None
        try:
            lbl = self.scene.add_3d_label(position.astype(float), str(idx))
        except Exception:
            pass
        self._markers.append((name, lbl))

    def undo_marker(self):
        if not self._markers:
            return
        name, lbl = self._markers.pop()
        try:
            self.scene.scene.remove_geometry(name)
        except Exception:
            pass
        try:
            if lbl is not None:
                self.scene.remove_3d_label(lbl)
        except Exception:
            pass

    # project all 3D points to widget pixel coords; return nearest index
    def pick_point_index(self, points_xyz: np.ndarray, click_xy: tuple[int, int], max_px_dist=12):
        cam = self.scene.scene.camera
        view = np.asarray(cam.get_view_matrix())          # 4x4
        proj = np.asarray(cam.get_projection_matrix())    # 4x4

        # world -> clip -> NDC -> screen
        N = points_xyz.shape[0]
        xyz1 = np.c_[points_xyz, np.ones((N, 1))]
        clip = (xyz1 @ view.T) @ proj.T
        w = clip[:, 3:4]
        valid = w[:, 0] > 1e-9
        clip = clip[valid]
        w = w[valid]
        if clip.size == 0:
            return None
        ndc = clip[:, :3] / w
        # visible (inside clip volume)
        vis = (np.abs(ndc[:, 0]) <= 1) & (np.abs(ndc[:, 1]) <= 1) & (ndc[:, 2] >= -1) & (ndc[:, 2] <= 1)
        if not np.any(vis):
            return None
        ndc = ndc[vis]

        # widget viewport
        r = self.scene.frame  # Rect(x, y, w, h)
        px = (ndc[:, 0] * 0.5 + 0.5) * r.width + r.x
        py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * r.height + r.y

        # nearest in screen pixels
        sx, sy = click_xy
        d2 = (px - sx) ** 2 + (py - sy) ** 2
        j = int(np.argmin(d2))
        if d2[j] <= (max_px_dist ** 2):
            # map back to original index
            idx_map = np.flatnonzero(valid)[vis]
            return int(idx_map[j])
        return None


class CorrespondenceTool(Node):
    def __init__(self):
        super().__init__('correspondence_tool')

        cfg = extract_configuration()
        self.data_dir = cfg['general']['data_folder']
        self.config_dir = cfg['general']['config_folder']
        self.out_txt  = os.path.join(self.data_dir, cfg['general']['correspondence_file'])
        intr_yaml     = os.path.join(self.config_dir,cfg['general']['camera_intrinsic_calibration'])
        self.extr_yaml= os.path.join(self.data_dir,"prior_calibration_extrinsics.yaml")
        self.K, self.D, (self.W, self.H) = utils.load_intrinsics(intr_yaml)

        self.pairs = self._scan_pairs(self.data_dir)
        if not self.pairs:
            self.get_logger().error(f"No .png/.pcd pairs found in '{self.data_dir}'")
            return

        # windows
        cv2.namedWindow(IMG_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(IMG_WIN, 1280, 720)
        cv2.setMouseCallback(IMG_WIN, self._on_mouse_image)

        self.o3d = PCDGui()

        # state
        self._img_clicks = []
        self._pcd_points = []
        self._img_overlay = None
        self._img_base = None
        self._pcd = None
        self._pcd_xyz = None

        self._go_next = False
        self._quit = False

        self.run()

    # ---------- IO ----------
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

    # ---------- image picking ----------
    def _on_mouse_image(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._img_clicks.append((int(x), int(y)))

    def _draw_image_overlay(self):
        show = self._img_base.copy()
        # header
        msg = "LMB=add, u=undo, s=save-next, f=calculate, q=quit"
        cv2.putText(show, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 230, 40), 2, cv2.LINE_AA)

        # points with numbers (red dot + black number centered)
        for i, (u, v) in enumerate(self._img_overlay_points, start=1):
            cv2.circle(show, (int(u), int(v)), 7, (0, 0, 255), -1)
            txt = str(i)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cx, cy = int(u) - tw // 2, int(v) + th // 2 - 2
            cv2.putText(show, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        return show

    def _update_prior_guess(self):
        calib.calibrate_extrinsics(extr_file=self.extr_yaml)
        if os.path.isfile(self.extr_yaml):
            try:
                self.T_guess = utils.load_extrinsic(self.extr_yaml)
                self.get_logger().info("Using existing extrinsic as guess for assisted 3D picking.")
            except Exception as e:
                self.get_logger().warn(f"Failed to load extrinsic guess: {e}")

    # ---------- main ----------
    def run(self):
        force_skip_mismatch = False
        force_skip_zero = False

        # # reset file
        # open(self.out_txt, 'w').close()

        image_index = 0
        while image_index < len(self.pairs):
            if not force_skip_mismatch and not force_skip_zero:
                name, img_path, pcd_path = self.pairs[image_index]
                image_index += 1

                if self._quit:
                    break

                img = cv2.imread(img_path)
                pcd = o3d.io.read_point_cloud(pcd_path)
                if img is None or pcd.is_empty():
                    self.get_logger().warn(f"Skip {name} (bad image/pcd).")
                    continue

                self._img_base = img
                self._img_overlay_points = []
                self._img_clicks.clear()

                self._pcd = pcd
                self._pcd_points = []
                self.o3d.set_pointcloud(pcd)
                self._pcd_xyz = np.asarray(pcd.points)

                cv2.setWindowTitle(IMG_WIN, f"Image: {name}")

            self._go_next = False
            while not self._go_next and not self._quit:
                # process Open3D events
                self.o3d.poll()

                # handle O3D mouse picks
                c = self.o3d.pop_click()
                if c is not None:
                    idx = self.o3d.pick_point_index(self._pcd_xyz, c, max_px_dist=PIX_EPS)
                    if idx is not None:
                        pt = self._pcd_xyz[idx]
                        self._pcd_points.append(pt.tolist())
                        self.o3d.add_marker(pt, len(self._pcd_points))

                # image pending clicks
                while self._img_clicks:
                    u, v = self._img_clicks.pop(0)
                    self._img_overlay_points.append((float(u), float(v)))

                # draw image overlay
                show = self._draw_image_overlay()
                cv2.imshow(IMG_WIN, show)

                # keys from OpenCV
                k = cv2.waitKey(10) & 0xFF
                if k:
                    self._handle_key(chr(k).lower())

                # keys from Open3D
                ok = self.o3d.pop_key()
                if ok:
                    self._handle_key(ok)

                # enforce paired operations visibility (optional)
                # (we let you select freely; we validate on save)

            # save/next if requested
            if self._quit:
                break

            number_p_img = len(self._img_overlay_points)
            number_p_pcd = len(self._pcd_points)
            if number_p_img == 0 and number_p_pcd == 0:
                if not force_skip_zero:
                    self.get_logger().info(f"No points for {name}; press 's' to force skip")
                    force_skip_zero = True
                else:
                    self.get_logger().info(f"No points for {name}; force skip")
                    force_skip_mismatch = False
                    force_skip_zero = False
                continue
            
            if number_p_img != number_p_pcd:
                if not force_skip_mismatch:
                    self.get_logger().warn(
                        f"Counts mismatch for {name}: image={number_p_img} vs pcd={number_p_pcd}; press 's' to force skip."
                    )
                    force_skip_mismatch = True
                else:
                    self.get_logger().warn(
                        f"Counts mismatch for {name}: image={number_p_img} vs pcd={number_p_pcd}; force skip."
                    )
                    force_skip_mismatch = False
                    force_skip_zero = False
                continue

            force_skip_mismatch = False
            force_skip_zero = False
            utils.write_corresp_append(self.out_txt, name, self._img_overlay_points, self._pcd_points)
            self.get_logger().info(f"Saved {number_p_pcd} pairs for {name} → {self.out_txt}")

        calib.calibrate_extrinsics(extr_file=self.extr_yaml)
        cv2.destroyWindow(IMG_WIN)

    def _handle_key(self, ch: str):
        if ch == 'u':
            # undo last pair atomically if both have elements; else extra point
            img_points, pcd_points = self._img_overlay_points, self._pcd_points
            d = len(img_points) - len(pcd_points)  # >0: extra img, <0: extra pcd, 0: pop both if present
            if img_points and d >= 0:
                img_points.pop()
            if pcd_points and d <= 0:
                pcd_points.pop()
                self.o3d.undo_marker()
        elif ch == 's':
            # move to next pair (save happens after loop with validation)
            self._go_next = True
        elif ch == 'f':
            self._update_prior_guess()
        elif ch == 'q':
            self._quit = True
        elif ch in ('[', '-'):
            self.o3d.set_point_size(self.o3d._pt_size - 1)
        elif ch in (']', '+', '='):
            self.o3d.set_point_size(self.o3d._pt_size + 1)
        # else ignore


def main(args=None):
    rclpy.init(args=args)
    node = CorrespondenceTool()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
