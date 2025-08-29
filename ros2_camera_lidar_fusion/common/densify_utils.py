
import numpy as np
import cv2

from ros2_camera_lidar_fusion.common import utils


def densify_velodyne_rings(
                        xyz_lidar: np.ndarray,
                        num_rings: int = 16,
                        az_bins: int = 2048,
                        interp_per_gap: int = 1,
                        max_range_jump: float = 1.5,
                        max_gap_m: float = 1.5) -> np.ndarray:
    """
    Densify a Velodyne-style scan by interpolating between adjacent rings inside azimuth bins.

    Steps:
    - Assign each 3D point to an azimuth bin and nearest vertical ring.
    - Select one representative point per (azimuth, ring) cell.
    - Identify neighboring valid cells along vertical rings.
    - Interpolate intermediate points between neighbors if:
        * range discontinuity is below `max_range_jump`
        * spatial gap is below `max_gap_m`
    - Number of interpolated points per gap grows constant with `interp_per_gap`.

    Args:
        xyz_lidar: (N, 3) points in LiDAR frame (float32 recommended).
        num_rings: Number of vertical rings (e.g., 16 for VLP-16).
        az_bins: Number of azimuth bins over [0, 2π).
        interp_per_gap: Max samples inserted between adjacent rings per azimuth bin.
        max_range_jump: Continuity gate; allow max curvature difference between rings. Set None to disable.
        max_gap_m: Max Euclidean gap allowed between neighbors. Set None to disable.

    Returns:
        Augmented point cloud (original + interpolated points), shape (N, 3).
    """
    if xyz_lidar.size == 0 or interp_per_gap <= 0:
        return xyz_lidar

    xyz = xyz_lidar.astype(np.float32, copy=False)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r_xy = np.hypot(x, y).astype(np.float32)
    rng = np.sqrt(r_xy * r_xy + z * z).astype(np.float32)

    # azimuth bins
    phi = np.arctan2(y, x).astype(np.float32)
    phi[phi < 0.0] += 2.0 * np.pi
    az_step = (2.0 * np.pi) / float(az_bins)
    az_bin = np.minimum((phi / az_step).astype(np.int32), az_bins - 1)

    # ring assignment (nearest VLP-16 vertical angles; sorted ascending)
    angles_deg = np.array([-15,-13,-11,-9,-7,-5,-3,-1, 1, 3, 5, 7, 9,11,13,15], dtype=np.float32)
    ang = (angles_deg * np.pi / 180.0).astype(np.float32)
    elev = np.arctan2(z, r_xy).astype(np.float32)
    rings = np.argmin(np.abs(elev[:, None] - ang[None, :]), axis=1).astype(np.int16)

    # representative per cell = (az_bin, ring) with minimal range
    cells = (az_bin.astype(np.int64) * num_rings + rings.astype(np.int64))
    order = np.lexsort((rng, cells))  # primary by cells, secondary by rng
    cells_sorted = cells[order]
    first_idx = np.concatenate(([0], np.flatnonzero(np.diff(cells_sorted)) + 1))
    rep_idx = order[first_idx]
    rep_cells = cells_sorted[first_idx]
    rep_xyz = xyz[rep_idx]
    rep_rng = rng[rep_idx]

    # grid [az, ring] -> representative point/range
    grid_xyz = np.full((az_bins, num_rings, 3), np.nan, dtype=np.float32)
    grid_rng = np.full((az_bins, num_rings), np.nan, dtype=np.float32)
    az_u = (rep_cells // num_rings).astype(np.int32)
    ring_u = (rep_cells % num_rings).astype(np.int32)
    grid_xyz[az_u, ring_u] = rep_xyz
    grid_rng[az_u, ring_u] = rep_rng

    # neighbors along rings
    S = grid_xyz[:, :-1, :]   # [az, ring-1, 3]
    E = grid_xyz[:,  1:, :]   # [az, ring  , 3]
    mask = np.isfinite(S[..., 0]) & np.isfinite(E[..., 0])

    if max_range_jump is not None:
        Sr = grid_rng[:, :-1]; Er = grid_rng[:, 1:]
        mnr = np.minimum(Sr, Er)
        mxr = np.maximum(Sr, Er)
        ratio = mxr / np.maximum(mnr, 1e-6)
        mask &= ratio <= float(max_range_jump)

    if max_gap_m is not None:
        d2 = np.sum((E - S) * (E - S), axis=2)
        mask &= d2 <= (float(max_gap_m) ** 2)

    if not np.any(mask):
        return xyz

    # vectorized interpolation
    pieces = []
    M = max(1, int(interp_per_gap))
    for t in range(1, M + 1):
        a = t / float(M + 1)
        P = (1.0 - a) * S + a * E
        P = P[mask]
        if P.size:
            pieces.append(P.astype(np.float32, copy=False))

    if not pieces:
        return xyz
    return np.vstack((xyz, np.concatenate(pieces, axis=0)))


def extract_lidar_image_params( xyz_lidar, cv_image, T_lidar_to_cam, camera_matrix, dist_coeffs):
    xyz_cam = utils.transform_points(T_lidar_to_cam, xyz_lidar)
    front = xyz_cam[:, 2] > 0.0
    xyz_cam = xyz_cam[front]

    # Project
    rvec = np.zeros((3,1)); tvec = np.zeros((3,1))
    uv, _ = cv2.projectPoints(xyz_cam, rvec, tvec, camera_matrix, dist_coeffs)
    uv = uv.reshape(-1, 2)

    h, w = cv_image.shape[:2]
    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # recover indices into original xyz_lidar
    keep_idx = np.flatnonzero(front)[in_img]
    mask = np.zeros(xyz_lidar.shape[0], dtype=bool)
    mask[keep_idx] = True

    return mask, u[in_img], v[in_img], xyz_cam[in_img, 2]

def filter_points_on_image(xyz_lidar, cv_image, T_lidar_to_cam, camera_matrix, dist_coeffs):
    mask, u, v, _ = extract_lidar_image_params(
        xyz_lidar, cv_image, T_lidar_to_cam, camera_matrix, dist_coeffs
    )

    dbg_img = cv_image.copy()
    for uu, vv in zip(u, v):
        cv2.circle(dbg_img, (int(uu), int(vv)), 1, (0, 255, 0), -1)

    return xyz_lidar[mask], dbg_img

def densify_depth_uniform( u, v, z, img, camera_matrix, grid_step_px=3, fill_iters=5, max_interp_dist=0.2):
    h, w = img.shape[:2]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    depth = np.full((h * w,), np.inf, np.float32)
    idx = (v.astype(np.int64) * w + u.astype(np.int64))
    np.minimum.at(depth, idx, z.astype(np.float32))
    depth = depth.reshape(h, w)

    for _ in range(max(1, fill_iters)):
        valid = np.isfinite(depth)
        num = cv2.blur(np.where(valid, depth, 0.0), (3, 3), borderType=cv2.BORDER_REPLICATE)
        den = cv2.blur(valid.astype(np.float32), (3, 3), borderType=cv2.BORDER_REPLICATE)
        fill_mask = ~valid & (den > 0)
        if not np.any(fill_mask):
            break
        avg_depth = num / np.maximum(den, 1e-6)
        min_d = cv2.erode(np.where(valid, depth, np.inf), np.ones((3, 3), np.uint8))
        max_d = cv2.dilate(np.where(valid, depth, -np.inf), np.ones((3, 3), np.uint8))
        ok = (max_d - min_d) < max_interp_dist
        final_mask = fill_mask & ok
        depth[final_mask] = avg_depth[final_mask]

    xs = np.arange(0, w, max(1, grid_step_px), dtype=np.int32)
    ys = np.arange(0, h, max(1, grid_step_px), dtype=np.int32)
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
