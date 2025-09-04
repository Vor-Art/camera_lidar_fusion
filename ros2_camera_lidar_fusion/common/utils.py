#!/usr/bin/env python3
import os, yaml, numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- IO ----------
def load_intrinsics(yaml_path):
    with open(yaml_path, 'r') as f: y = yaml.safe_load(f)
    K = np.array(y['camera_matrix']['data'], float).reshape(3,3)
    D = np.array(y['distortion_coefficients']['data'], float).reshape(1,-1)
    W = int(y['image_size']['width'])
    H = int(y['image_size']['height'])
    return K, D, (W,H)

def load_extrinsic(yaml_path):
    with open(yaml_path,'r') as f: y = yaml.safe_load(f)
    T = np.array(y['transformation_matrix'], float)
    assert T.shape==(4,4)
    return T

def save_extrinsic_aligned(T, out_yaml):
    R = T[:3,:3]; t = T[:3,3]
    with open(out_yaml,'w') as f:
        f.write("# (LiDAR -> Camera)\n")
        f.write("translation: [{: .8f}, {: .8f}, {: .8f}]\n".format(*t))
        rflat = R.reshape(-1)
        f.write("rotation:    [")
        for i,v in enumerate(rflat):
            f.write(f"{v: .8f}")
            if i < len(rflat)-1: f.write(", ")
            if (i+1)%3==0 and i < len(rflat)-1: f.write("\n              ")
        f.write("]\n")
        f.write("transformation_matrix:\n")
        for row in T:
            f.write("  - [{}]\n".format(", ".join(f"{v: .8f}" for v in row)))

# ---------- Geometry ----------
def transform_points(T, pts3d):
    N = len(pts3d)
    ph = np.hstack([pts3d, np.ones((N,1))])
    qc = (ph @ T.T)[:,:3]
    return qc

def invert_h(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3]  = -R.T @ t
    return Ti

def project_points(pts_cam, K, D):
    rvec = np.zeros((3,1)); tvec = np.zeros((3,1))
    uv,_ = cv2.projectPoints(pts_cam.astype(np.float64), rvec, tvec, K, D)
    return uv.reshape(-1,2)

# ---------- Robust PnP ----------
def pnp_ransac_refine(pts3d, pts2d, K, D,
                      ransac_px=6.0, conf=0.999, iters=5000,
                      min_inliers=6, guess_T=None):
    """
    Robust PnP:
      - sanitizes data
      - tries multiple solvers (EPNP/AP3P/SQPNP) with RANSAC
      - optional warm-start from guess_T
      - LM refine on inliers
      - fallback: randomized minimal subsets if RANSAC fails
    Returns:
      T (4x4), inlier_idx (int array)
    """
    pts2d = np.asarray(pts2d, dtype=np.float64).reshape(-1, 2)
    pts3d = np.asarray(pts3d, dtype=np.float64).reshape(-1, 3)

    # 0) sanitize
    if pts2d.shape[0] != pts3d.shape[0]:
        raise ValueError(f"2D/3D size mismatch: {pts2d.shape[0]} vs {pts3d.shape[0]}")
    N0 = pts2d.shape[0]
    mask_finite = np.isfinite(pts2d).all(axis=1) & np.isfinite(pts3d).all(axis=1)
    pts2d, pts3d = pts2d[mask_finite], pts3d[mask_finite]
    if pts2d.shape[0] < 4:
        raise RuntimeError("Not enough finite correspondences (need >=4)")

    # dedupe very-close duplicates (helps manual clicks)
    def _dedupe(a, eps):
        k = np.round(a / eps).astype(np.int64)
        _, idx = np.unique(k, axis=0, return_index=True)
        return idx
    keep_uv = _dedupe(pts2d, eps=0.5)  # 0.5 px grid
    pts2d = pts2d[keep_uv]; pts3d = pts3d[keep_uv]
    if pts2d.shape[0] < 4:
        raise RuntimeError("Too many duplicates removed; need >=4 unique points")

    # distortion handling
    D_use = None
    if D is not None:
        D = np.asarray(D, dtype=np.float64).reshape(1, -1)
        if D.size > 0 and np.any(np.abs(D) > 1e-12):
            D_use = D

    # optional warm start
    use_guess = False
    rvec0 = None; tvec0 = None
    if guess_T is not None:
        R0 = guess_T[:3, :3]; t0 = guess_T[:3, 3]
        rvec0, _ = cv2.Rodrigues(R0.astype(np.float64))
        tvec0 = t0.reshape(3, 1).astype(np.float64)
        use_guess = True

    # 1) try RANSAC with several solvers
    flags_try = [cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_AP3P]
    if hasattr(cv2, "SOLVEPNP_SQPNP"):
        flags_try.insert(0, cv2.SOLVEPNP_SQPNP)

    def _run_ransac(flag, reproj_px):
        return cv2.solvePnPRansac(pts3d, pts2d, K, D_use,
                                  rvec=rvec0 if use_guess else None,
                                  tvec=tvec0 if use_guess else None,
                                  useExtrinsicGuess=use_guess,
                                  iterationsCount=int(iters),
                                  reprojectionError=float(reproj_px),
                                  confidence=float(conf),
                                  flags=flag)

    ok = False
    best = None  # will hold (rvec, tvec, inliers)

    for reproj_px in [ransac_px, max(8.0, ransac_px), max(12.0, ransac_px)]:
        for flag in flags_try:
            ret = _run_ransac(flag, reproj_px)  # (ok_ret, rvec, tvec, inliers)
            ok_ret, rvec, tvec, inliers = ret
            if ok_ret and inliers is not None and len(inliers) >= min_inliers:
                ok = True
                best = (rvec, tvec, inliers)
                break
        if ok:
            break

    if ok:
        rvec, tvec, inliers = best
        inliers = np.asarray(inliers).ravel().astype(int)
        rvec, tvec = cv2.solvePnPRefineLM(pts3d[inliers], pts2d[inliers], K, D_use, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = tvec.ravel()
        return T, inliers

    # 2) fallback: randomized minimal subsets
    rng = np.random.default_rng(12345)
    best_err = np.inf; best_T = None; best_inliers = None
    M = min(300, max(100, 30 * min(10, pts2d.shape[0])))  # attempts
    for _ in range(M):
        if pts2d.shape[0] >= 6:
            idx = rng.choice(pts2d.shape[0], size=6, replace=False)
            flag = cv2.SOLVEPNP_EPNP
        else:
            idx = rng.choice(pts2d.shape[0], size=4, replace=False)
            flag = cv2.SOLVEPNP_AP3P
        ok2, rvec2, tvec2 = cv2.solvePnP(pts3d[idx], pts2d[idx], K, D_use, flags=flag)
        if not ok2: continue
        R2, _ = cv2.Rodrigues(rvec2)
        T2 = np.eye(4); T2[:3, :3] = R2; T2[:3, 3] = tvec2.ravel()
        res, _ = residuals_px(T2, pts3d, pts2d, K, D_use if D_use is not None else np.zeros((1, 0)))
        med = np.median(res); mad = np.median(np.abs(res - med)) + 1e-9
        thr = med + 3.0 * 1.4826 * mad
        inl = np.where(res <= max(thr, ransac_px))[0]
        if inl.size >= min_inliers:
            rvec2, tvec2 = cv2.solvePnPRefineLM(pts3d[inl], pts2d[inl], K, D_use, rvec2, tvec2)
            R2, _ = cv2.Rodrigues(rvec2)
            T2[:3, :3] = R2; T2[:3, 3] = tvec2.ravel()
            res2, _ = residuals_px(T2, pts3d, pts2d, K, D_use if D_use is not None else np.zeros((1, 0)))
            err = np.median(res2)
            if err < best_err:
                best_err = err; best_T = T2.copy(); best_inliers = inl

    if best_T is not None:
        return best_T, best_inliers

    raise RuntimeError("PnP failed: data too degenerate or thresholds too strict.")

def residuals_px(T, pts3d, pts2d, K, D):
    pc = transform_points(T, pts3d)
    uv = project_points(pc, K, D)
    res = np.linalg.norm(uv - pts2d, axis=1)
    return res, uv

def mad_trimming(res, k=3.0):
    med = np.median(res)
    mad = np.median(np.abs(res - med)) + 1e-9
    sigma = 1.4826 * mad
    keep = np.abs(res - med) <= k * sigma
    return keep

# ---------- Plots ----------
def plot_residuals(res, out_dir, tag="all", bins=40):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(); plt.hist(res, bins=bins); plt.xlabel("reproj error [px]"); plt.ylabel("count")
    plt.title(f"Residual histogram ({tag})"); plt.grid(True, ls=':')
    plt.savefig(os.path.join(out_dir, f"residual_hist_{tag}.png"), dpi=160); plt.close()

    plt.figure(); s=np.sort(res); c=np.linspace(0,1,len(s))
    plt.plot(s,c); plt.xlabel("reproj error [px]"); plt.ylabel("CDF")
    plt.title(f"Residual CDF ({tag})"); plt.grid(True, ls=':')
    plt.savefig(os.path.join(out_dir, f"residual_cdf_{tag}.png"), dpi=160); plt.close()

    plt.figure(); plt.boxplot(res, vert=True, labels=[tag])
    plt.ylabel("reproj error [px]"); plt.grid(True, ls=':')
    plt.savefig(os.path.join(out_dir, f"residual_box_{tag}.png"), dpi=160); plt.close()

def plot_quiver(width, height, pts2d, proj2d, out_path, step=1):
    canvas = np.ones((height, width, 3), np.uint8)*255
    P = pts2d[::step]; Q = proj2d[::step]
    U = (Q[:,0]-P[:,0]); V=(Q[:,1]-P[:,1])
    plt.figure(figsize=(width/120, height/120), dpi=120)
    plt.imshow(canvas[:,:,::-1])
    plt.quiver(P[:,0], P[:,1], U, V, np.hypot(U,V), angles='xy', scale_units='xy', scale=1, cmap='viridis')
    plt.gca().invert_yaxis(); plt.axis('off')
    plt.tight_layout(pad=0); plt.savefig(out_path, dpi=120); plt.close()

# ---------- Correspondence IO ----------
def read_correspondences(txt_path):
    pts2d=[]; pts3d=[]
    with open(txt_path,'r') as f:
        for line in f:
            s=line.strip()
            if not s or s.startswith('#'): continue
            t=s.split(',')
            if len(t)!=5: continue
            u,v,X,Y,Z = map(float,t)
            pts2d.append([u,v]); pts3d.append([X,Y,Z])
    return np.array(pts2d,float), np.array(pts3d,float)

def write_corresp_append(path, pair_name, uv, xyz):
    with open(path,'a') as f:
        f.write(f"# Pair: {pair_name}\n# u, v, x, y, z\n")
        for (u,v),(X,Y,Z) in zip(uv,xyz):
            f.write(f"{u},{v},{X},{Y},{Z}\n")
        f.write("\n")

# ---------- ROS helpers ----------
def pointcloud2_to_xyz_array_fast(msg, skip_rate=1):
    if getattr(msg, 'height', 0) == 0 or getattr(msg, 'width', 0) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    fields = [f.name for f in msg.fields]
    if not all(k in fields for k in ('x','y','z')):
        return np.zeros((0, 3), dtype=np.float32)
    dtype = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('_', f'V{msg.point_step - 12}')
    ])
    raw = np.frombuffer(msg.data, dtype=dtype)
    pts = np.vstack((raw['x'], raw['y'], raw['z'])).T
    return pts[::skip_rate] if skip_rate > 1 else pts

def voxel_downsample_with_color(xyz_lidar, px, img_bgr, voxel_size):
    if xyz_lidar.shape[0] == 0:
        return np.zeros((0,3), np.float32), np.zeros((0,), np.float32)
    bgr = img_bgr[px[:,1], px[:,0], :]
    r = bgr[:,2].astype(np.float32); g = bgr[:,1].astype(np.float32); b = bgr[:,0].astype(np.float32)
    vs = float(voxel_size)
    keys = np.floor(xyz_lidar / vs).astype(np.int32)
    key_view = keys.view([('ix', np.int32), ('iy', np.int32), ('iz', np.int32)]).reshape(-1)
    _, inv = np.unique(key_view, return_inverse=True)
    cnt = np.bincount(inv)
    sx = np.bincount(inv, weights=xyz_lidar[:,0])
    sy = np.bincount(inv, weights=xyz_lidar[:,1])
    sz = np.bincount(inv, weights=xyz_lidar[:,2])
    sr = np.bincount(inv, weights=r); sg = np.bincount(inv, weights=g); sb = np.bincount(inv, weights=b)
    xyz_ds = np.stack((sx/cnt, sy/cnt, sz/cnt), axis=1).astype(np.float32)
    r_avg = (sr/cnt).astype(np.uint32); g_avg = (sg/cnt).astype(np.uint32); b_avg = (sb/cnt).astype(np.uint32)
    rgb_u32 = (r_avg << 16) | (g_avg << 8) | b_avg
    rgb_f32 = rgb_u32.view(np.float32)
    return xyz_ds, rgb_f32


def voxelize_numpy(points: np.ndarray, voxel_size: float) -> np.ndarray:
    return voxel_centroids_ravel_bincount(points, voxel_size)

def voxel_centroids_sort_reduce(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0: return points
    vs = float(voxel_size)
    grid = np.floor(points / vs).astype(np.int64)
    order = np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0]))
    g = grid[order]
    p = points[order].astype(np.float64)
    change = np.any(np.diff(g, axis=0) != 0, axis=1)
    idx = np.concatenate(([True], change)).nonzero()[0]
    counts = np.diff(np.append(idx, g.shape[0]))
    sums = np.add.reduceat(p, idx, axis=0)
    return (sums / counts[:, None]).astype(np.float32)

def voxel_centroids_ravel_bincount(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0: return points
    vs = float(voxel_size)
    grid = np.floor(points / vs).astype(np.int64)
    gmin = grid.min(axis=0)
    grid -= gmin
    span = grid.max(axis=0) + 1
    cap = np.iinfo(np.int64).max
    if int(span[0]) * int(span[1]) * int(span[2]) >= cap:
        return voxel_centroids_sort_reduce(points, voxel_size)
    lin = grid[:, 0] + span[0] * (grid[:, 1] + span[1] * grid[:, 2])
    lin = lin.astype(np.int64)
    uniq, inv, counts = np.unique(lin, return_inverse=True, return_counts=True)
    x = np.bincount(inv, weights=points[:, 0].astype(np.float64), minlength=uniq.shape[0])
    y = np.bincount(inv, weights=points[:, 1].astype(np.float64), minlength=uniq.shape[0])
    z = np.bincount(inv, weights=points[:, 2].astype(np.float64), minlength=uniq.shape[0])
    centers = np.stack((x, y, z), axis=1) / counts[:, None]
    return centers.astype(np.float32)

# ---------- New: bit-pack + sort + single reduc

def voxel_downsample_with_color(xyz_lidar: np.ndarray,
                                px: np.ndarray,
                                img_bgr: np.ndarray,
                                voxel_size: float):
    # Empty guard
    if xyz_lidar.shape[0] == 0 or px.shape[0] == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)

    # Keep arrays aligned (defensive)
    n = min(xyz_lidar.shape[0], px.shape[0])
    if n == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)
    xyz = np.ascontiguousarray(xyz_lidar[:n], dtype=np.float32)
    u = np.ascontiguousarray(px[:n, 0], dtype=np.int64)
    v = np.ascontiguousarray(px[:n, 1], dtype=np.int64)

    # 1) Gather BGR with flat indexing (fewer fancy-index ops)
    h, w = img_bgr.shape[:2]
    lin_px = v * w + u
    bgr = img_bgr.reshape(-1, 3)[lin_px]  # uint8
    r = bgr[:, 2].astype(np.float64)  # sum in float64 for accuracy
    g = bgr[:, 1].astype(np.float64)
    b = bgr[:, 0].astype(np.float64)

    # 2) Build integer grid & linear voxel keys (ravel+bincount path)
    vs = float(voxel_size)
    inv_vs = 1.0 / vs
    grid = np.floor(xyz * inv_vs).astype(np.int64)  # int grid
    gmin = grid.min(axis=0); grid -= gmin
    span = grid.max(axis=0) + 1

    # Overflow guard: if span product too large, fall back to sort+reduce
    cap = np.iinfo(np.int64).max // 4
    if int(span[0]) * int(span[1]) * int(span[2]) >= cap:
        return _voxel_downsample_with_color_sort_reduce(xyz, r, g, b, voxel_size)

    lin = grid[:, 0] + span[0] * (grid[:, 1] + span[1] * grid[:, 2])
    # 3) Unique once → inv map and counts
    uniq, inv, counts = np.unique(lin, return_inverse=True, return_counts=True)
    m = uniq.shape[0]
    denom = counts.astype(np.float64)

    # 4) Accumulate xyz and colors with bincount (minlength=m ensures shape)
    x_sum = np.bincount(inv, weights=xyz[:, 0].astype(np.float64), minlength=m)
    y_sum = np.bincount(inv, weights=xyz[:, 1].astype(np.float64), minlength=m)
    z_sum = np.bincount(inv, weights=xyz[:, 2].astype(np.float64), minlength=m)
    r_sum = np.bincount(inv, weights=r, minlength=m)
    g_sum = np.bincount(inv, weights=g, minlength=m)
    b_sum = np.bincount(inv, weights=b, minlength=m)

    xyz_ds = np.stack((x_sum/denom, y_sum/denom, z_sum/denom), axis=1).astype(np.float32)

    # 5) Average BGR → pack to float32 RGB as in PCL convention
    r_avg = np.rint(r_sum/denom).astype(np.uint32)
    g_avg = np.rint(g_sum/denom).astype(np.uint32)
    b_avg = np.rint(b_sum/denom).astype(np.uint32)
    rgb_u32 = (r_avg << 16) | (g_avg << 8) | b_avg
    rgb_f32 = rgb_u32.view(np.float32)
    return xyz_ds, rgb_f32


def _voxel_downsample_with_color_sort_reduce(xyz: np.ndarray,
                                             r: np.ndarray, g: np.ndarray, b: np.ndarray,
                                             voxel_size: float):
    """Fallback when ravel space would overflow: bit-pack + sort + reduceat."""
    if xyz.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)
    vs = float(voxel_size); inv_vs = 1.0 / vs
    grid = np.floor(xyz * inv_vs).astype(np.int32)
    grid -= grid.min(axis=0)

    # 21-bit packing per axis: up to ~2 million voxels per axis
    span = grid.max(axis=0) + 1
    if np.any(span >= (1 << 21)):
        # if even 21 bits per axis is too small, fall back to lexsort reduce
        order = np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0]))
        key_sorted = grid[order]
        packed = np.any(np.diff(key_sorted, axis=0) != 0, axis=1)
        starts = np.concatenate(([True], packed))
        idx = np.flatnonzero(starts)
        counts = np.diff(np.append(idx, key_sorted.shape[0]))
        P = np.c_[xyz[order].astype(np.float64),
                  r[order], g[order], b[order]]
        sums = np.add.reduceat(P, idx, axis=0)
    else:
        key = (grid[:, 0].astype(np.uint64) << 42) | \
              (grid[:, 1].astype(np.uint64) << 21) | \
               grid[:, 2].astype(np.uint64)
        order = np.argsort(key)
        key_s = key[order]
        starts = np.empty(key_s.size, dtype=bool)
        starts[0] = True
        np.not_equal(key_s[1:], key_s[:-1], out=starts[1:])
        idx = np.flatnonzero(starts)
        counts = np.diff(np.append(idx, key_s.size))
        P = np.c_[xyz[order].astype(np.float64),
                  r[order], g[order], b[order]]
        sums = np.add.reduceat(P, idx, axis=0)

    denom = counts.astype(np.float64)[:, None]
    means = (sums / denom)
    xyz_ds = means[:, :3].astype(np.float32)
    r_avg = np.rint(means[:, 3]).astype(np.uint32)
    g_avg = np.rint(means[:, 4]).astype(np.uint32)
    b_avg = np.rint(means[:, 5]).astype(np.uint32)
    rgb_u32 = (r_avg << 16) | (g_avg << 8) | b_avg
    rgb_f32 = rgb_u32.view(np.float32)
    return xyz_ds, rgb_f32