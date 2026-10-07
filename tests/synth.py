"""Synthetic camera images with known ground truth, rendered with OpenCV only.

A plane texture is warped into the image with H = K [r1 r2 t], then lens distortion is applied
by remapping, so calibration and pose code can be checked against exact numbers.
"""
import cv2
import numpy as np


def make_K(w=1280, h=720, f=900.0, dcx=3.0, dcy=-2.0):
    return np.array([[f, 0, w / 2 + dcx], [0, f, h / 2 + dcy], [0, 0, 1.0]])


def rodrigues_R(rvec):
    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


def look_at(cam_pos, target, world_down):
    """T_cam_world (4x4) for a camera at cam_pos looking at target, with image-down along world_down.

    OpenCV camera axes: x right, y down, z forward.
    """
    cam_pos, target, world_down = (np.asarray(v, float) for v in (cam_pos, target, world_down))
    z = target - cam_pos
    z /= np.linalg.norm(z)
    y = world_down - world_down.dot(z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R_wc = np.stack([x, y, z], axis=1)           # camera axes in world coordinates
    T = np.eye(4)
    T[:3, :3] = R_wc.T
    T[:3, 3] = -R_wc.T @ cam_pos
    return T


def distortion_maps(K, dist, size):
    """cv2.remap maps that turn an ideal pinhole render into one with lens distortion ``dist``."""
    w, h = size
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([uu.ravel(), vv.ravel()], axis=1).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(pts, K, np.asarray(dist, float), P=K)   # distorted pixel -> ideal pixel
    ideal = ideal.reshape(h, w, 2)
    return ideal[..., 0].astype(np.float32), ideal[..., 1].astype(np.float32)


def warp_plane(texture, A_tex_to_plane, T_cam_plane, K, size, ss=3):
    """Warp a plane texture into the image. Returns (image, coverage mask).

    A_tex_to_plane maps texture pixel indices (u, v, 1) to plane coordinates (X, Y, 1) in metres;
    T_cam_plane is the pose of the plane's frame in the camera (plane is z = 0 in its own frame).
    The warp point-samples, which aliases hard edges by up to half a pixel, so it renders ``ss`` times
    larger and area-averages down. OpenCV puts pixel centres at integer coordinates, hence the
    (ss - 1) / 2 term when scaling K.
    """
    S = np.array([[ss, 0, (ss - 1) / 2], [0, ss, (ss - 1) / 2], [0, 0, 1.0]])
    H = S @ K @ np.column_stack([T_cam_plane[:3, 0], T_cam_plane[:3, 1], T_cam_plane[:3, 3]]) @ A_tex_to_plane
    big = (size[0] * ss, size[1] * ss)
    img = cv2.warpPerspective(texture, H, big, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    mask = cv2.warpPerspective(np.full(texture.shape[:2], 255, np.uint8), H, big, flags=cv2.INTER_NEAREST)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.resize(mask, size, interpolation=cv2.INTER_AREA)


def finish(img, K, dist, size, blur=0.8, noise=0.0, seed=0):
    """Lens distortion, optical softness and sensor noise."""
    if dist is not None and np.any(dist):
        mx, my = distortion_maps(K, dist, size)
        img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise:
        rng = np.random.default_rng(seed)
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def checkerboard_texture(pattern, ppu=60):
    """Checkerboard with a one-square white margin. Returns (texture, texture->plane matrix for unit squares).

    pattern is the inner-corner count (cols, rows), so the board has (cols+1) x (rows+1) squares.
    The plane origin is the first inner corner; +x along columns, +y along rows.
    """
    cols, rows = pattern
    n_c, n_r = cols + 1, rows + 1
    tex = np.full(((n_r + 2) * ppu, (n_c + 2) * ppu), 255, np.uint8)
    for r in range(n_r):
        for c in range(n_c):
            if (r + c) % 2 == 0:
                tex[(r + 1) * ppu:(r + 2) * ppu, (c + 1) * ppu:(c + 2) * ppu] = 0
    # pixel (u, v) -> squares: (u/ppu - 2, v/ppu - 2) puts the first inner corner at the origin
    A = np.array([[1 / ppu, 0, -2.0 + 0.5 / ppu], [0, 1 / ppu, -2.0 + 0.5 / ppu], [0, 0, 1.0]])
    return tex, A


def render_checkerboard(K, dist, T_cam_board, size, pattern=(9, 6), square=0.025, **finish_kw):
    tex, A_unit = checkerboard_texture(pattern)
    A = np.diag([square, square, 1.0]) @ A_unit        # squares -> metres
    img, _ = warp_plane(tex, A, T_cam_board, K, size)
    return finish(img, K, dist, size, **finish_kw)


def checkerboard_in_view(K, T_cam_board, size, pattern=(9, 6), square=0.025, margin=30):
    """True when every inner corner and the outer squares are inside the image by ``margin`` px."""
    cols, rows = pattern
    ext = np.array([[x, y, 0] for x in (-square, cols * square) for y in (-square, rows * square)] +
                   [[c * square, r * square, 0] for c in range(cols) for r in range(rows)], float)
    cam = (T_cam_board[:3, :3] @ ext.T).T + T_cam_board[:3, 3]
    if np.any(cam[:, 2] <= 0.05):
        return False
    pix = (K @ cam.T).T
    pix = pix[:, :2] / pix[:, 2:3]
    return bool(np.all(pix > margin) and np.all(pix[:, 0] < size[0] - margin) and np.all(pix[:, 1] < size[1] - margin))


def marker_texture(dictionary, marker_id, ppm, quiet_mm=10.0, side_mm=60.0):
    """ArUco marker (black border included in ``side_mm``) on a white quiet zone.

    Returns (texture, texture->plane matrix in metres). The plane frame is the marker's own frame:
    origin at the marker centre, x right, y UP (OpenCV's marker/solvePnP convention), z towards the viewer.
    """
    side_px = int(round(side_mm * ppm))
    quiet_px = int(round(quiet_mm * ppm))
    m = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px, borderBits=1)
    tex = np.full((side_px + 2 * quiet_px,) * 2, 255, np.uint8)
    tex[quiet_px:quiet_px + side_px, quiet_px:quiet_px + side_px] = m
    c = tex.shape[0] / 2
    A = np.array([[1 / (ppm * 1000), 0, (0.5 - c) / (ppm * 1000)], [0, -1 / (ppm * 1000), (c - 0.5) / (ppm * 1000)], [0, 0, 1.0]])
    return tex, A


def render_scene(K, dist, T_cam_board, size, marker_side, gap, cube_marker_side, cube_xy_yaw, cube_height,
                 dictionary=None, ids=(0, 1, 2, 3), cube_id=10, with_cube=True, **finish_kw):
    """The 2x2 GridBoard on the table plus a cube marker on top of a cube.

    Board frame is OpenCV's GridBoard frame: origin at marker 0's corner, x right, y down, z into
    the table. The cube marker lies in the plane z = -cube_height at planar position and yaw
    ``cube_xy_yaw`` = (x, y, yaw) in the board frame; yaw is the angle of the marker's x axis about board z.
    Returns the image and the ground-truth T_board_marker.
    """
    dictionary = dictionary or cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    ppm = 4   # texture pixels per mm (0.125 mm placement quantisation); marker is ~1.5x oversampled
    page = np.full((int(0.30 * ppm * 1000),) * 2, 255, np.uint8)       # 300 mm sheet, origin 70 mm in
    origin_mm = 70
    S, G = marker_side * 1000, gap * 1000
    pitch = S + G
    for i, mid in enumerate(ids):
        m = cv2.aruco.generateImageMarker(dictionary, mid, int(round(S * ppm)), borderBits=1)
        x0 = int(round((origin_mm + (i % 2) * pitch) * ppm))
        y0 = int(round((origin_mm + (i // 2) * pitch) * ppm))
        page[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    o = (0.5 / ppm - origin_mm) / 1000   # pixel centres sit half a pixel inside the texel edge
    A_page = np.array([[1 / (ppm * 1000), 0, o], [0, 1 / (ppm * 1000), o], [0, 0, 1.0]])
    img, _ = warp_plane(page, A_page, T_cam_board, K, size)

    x, y, yaw = cube_xy_yaw
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]) @ np.diag([1, -1, -1])
    T_board_marker = np.eye(4)
    T_board_marker[:3, :3] = R
    T_board_marker[:3, 3] = [x, y, -cube_height]
    if with_cube:
        tex, A = marker_texture(dictionary, cube_id, ppm, quiet_mm=8.0, side_mm=cube_marker_side * 1000)
        cube_img, mask = warp_plane(tex, A, T_cam_board @ T_board_marker, K, size)
        img = np.where(mask > 127, cube_img, img)
    return finish(img, K, dist, size, **finish_kw), T_board_marker
