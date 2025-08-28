#!/usr/bin/env python3
import os, numpy as np, yaml, cv2, rclpy
from rclpy.node import Node
from ros2_camera_lidar_fusion.common.read_yaml import extract_configuration
from ros2_camera_lidar_fusion.common import utils

class CalibrateExtrinsics(Node):
    def __init__(self):
        super().__init__('calibrate_extrinsics')
        cfg = extract_configuration()
        cfg_dir = cfg['general']['config_folder']
        data_dir= cfg['general']['data_folder']
        corr    = os.path.join(data_dir, cfg['general']['correspondence_file'])
        intr    = os.path.join(cfg_dir,  cfg['general']['camera_intrinsic_calibration'])
        outyml  = os.path.join(cfg_dir,  cfg['general']['camera_extrinsic_calibration'])
        plotdir = os.path.join(data_dir,  "extrinsic_plots")
        os.makedirs(plotdir, exist_ok=True)

        K, D, (W, H) = utils.load_intrinsics(intr)
        pts2d, pts3d = utils.read_correspondences(corr)
        if len(pts2d) < 8:
            raise RuntimeError("Need >= 8 correspondences for robust solve")

        # try strict first, then relax
        trials = [
            dict(ransac_px=3.0, iters=5000),
            dict(ransac_px=6.0, iters=7000),
            dict(ransac_px=10.0, iters=9000),
        ]
        last_err = None
        for tr in trials:
            try:
                print("N:", len(pts2d),
                        "uv range:", pts2d.min(0), pts2d.max(0),
                        "xyz norm median:", np.median(np.linalg.norm(pts3d, axis=1)))

                T, inl = utils.pnp_ransac_refine(pts3d, pts2d, K, D, **tr)
                break
            except Exception as e:
                last_err = e
        else:
            raise last_err

        # residuals + trimming as before...
        res_all, proj_all = utils.residuals_px(T, pts3d, pts2d, K, D)
        keep = utils.mad_trimming(res_all, k=3.0)
        if keep.sum() >= 6:
            T, inl2 = utils.pnp_ransac_refine(pts3d[keep], pts2d[keep], K, D,
                                            ransac_px=2.5, iters=5000)
            res_all, proj_all = utils.residuals_px(T, pts3d, pts2d, K, D)

        utils.save_extrinsic_aligned(T, outyml)
        # plots as you had
        self.get_logger().info(f"Saved extrinsics → {outyml}")

        # plots
        utils.plot_residuals(res_all, plotdir, tag="final")
        utils.plot_quiver(W, H, pts2d, proj_all, os.path.join(plotdir,"residual_quiver.png"))

        # dump inlier/outlier lists
        idx = np.arange(len(pts2d))
        inliers = idx[res_all <= np.median(res_all)+3*1.4826*np.median(np.abs(res_all-np.median(res_all)))]
        outliers = np.setdiff1d(idx, inliers)
        np.savetxt(os.path.join(plotdir,"inliers.txt"), inliers, fmt="%d")
        np.savetxt(os.path.join(plotdir,"outliers.txt"), outliers, fmt="%d")

def main(args=None):
    rclpy.init(args=args)
    n = CalibrateExtrinsics()
    rclpy.spin_once(n, timeout_sec=0.1)
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
