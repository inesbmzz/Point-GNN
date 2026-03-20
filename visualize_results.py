"""Visualize Point-GNN detections from saved results without re-running inference.

Self-contained: does NOT import open3d or tensorflow — safe on headless HPC nodes.

Supports three output modes:
  --mode ply   : saves a .ply point cloud per frame → open in CloudCompare
  --mode html  : saves an interactive 3D HTML per frame → open in any browser
  --mode bev   : saves a bird's-eye-view PNG per frame → open anywhere
"""

import os
import argparse
import numpy as np

# ── colors (R, G, B) in [0, 1] ────────────────────────────────────────────────
GT_COLOR = np.array([0.0, 1.0, 0.0])

CLASS_COLORS = {
    'Car':        np.array([1.0, 0.15, 0.15]),
    'Pedestrian': np.array([0.0, 0.47,  1.0]),
    'Cyclist':    np.array([1.0, 0.65,  0.0]),
}
DEFAULT_PRED_COLOR = np.array([1.0, 0.0, 0.0])

# ── argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Export Point-GNN result visualizations (no display needed).')
parser.add_argument('results_dir',
    help='Path to saved results dir (contains data/*.txt files).')
parser.add_argument('--dataset_root_dir', default='../dataset/kitti/',
    help='Path to KITTI dataset root.')
parser.add_argument('--dataset_split_file', default='',
    help='Split file. Defaults to 3DOP_splits/val.txt inside dataset_root_dir.')
parser.add_argument('--frame', type=int, default=0,
    help='Frame index to export (default=0). Use -1 to export all frames.')
parser.add_argument('--score_thresh', type=float, default=0.1,
    help='Minimum detection score to display (default=0.1).')
parser.add_argument('--mode', default='html', choices=['ply', 'html', 'bev', 'image'],
    help='Output format (default=html).')
parser.add_argument('--output_dir', default='',
    help='Where to save files. Defaults to results_dir/viz/')
parser.add_argument('--no-lidar', dest='show_lidar', action='store_false', default=True,
    help='Hide LiDAR point overlay on image mode.')
parser.add_argument('--no-gt', dest='show_gt', action='store_false', default=True,
    help='Hide ground truth boxes.')
parser.add_argument('--no-pred', dest='show_pred', action='store_false', default=True,
    help='Hide predicted boxes.')
args = parser.parse_args()

DATASET_DIR  = args.dataset_root_dir
RESULTS_DIR  = args.results_dir
SCORE_THRESH = args.score_thresh
SPLIT_FILE   = args.dataset_split_file or os.path.join(DATASET_DIR, '3DOP_splits/val.txt')
OUTPUT_DIR   = args.output_dir or os.path.join(RESULTS_DIR, 'viz')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── minimal KITTI helpers (no open3d dependency) ───────────────────────────────

def read_split_file(split_file):
    with open(split_file) as f:
        return [l.strip().split('.')[0] for l in f if l.strip()]


def load_calib(calib_path):
    """Parse a KITTI calibration file and return the velo→cam transform."""
    calib = {}
    with open(calib_path) as f:
        for line in f:
            fields = line.split()
            if len(fields) < 2:
                continue
            calib[fields[0].rstrip(':')] = np.array(fields[1:], dtype=np.float32)
    P2            = calib['P2'].reshape(3, 4)
    R0_rect       = calib['R0_rect'].reshape(3, 3)
    Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)

    R0_rect_4x4 = np.eye(4)
    R0_rect_4x4[:3, :3] = R0_rect
    Tr_4x4 = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    velo_to_rect = R0_rect_4x4 @ Tr_4x4   # (4,4)

    rect_to_cam = np.hstack([R0_rect,
                              np.linalg.inv(P2[:, :3]) @ P2[:, [3]]])
    rect_to_cam = np.vstack([rect_to_cam, [0, 0, 0, 1]])
    velo_to_cam = rect_to_cam @ velo_to_rect

    cam_to_image = np.hstack([P2[:, :3], np.zeros((3, 1))])  # (3,4)
    return {'velo_to_cam': velo_to_cam, 'cam_to_image': cam_to_image, 'P2': P2}


def load_velo_points(bin_path):
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3], pts[:, [3]]   # xyz, intensity


def velo_to_cam(pts_xyz, calib):
    n = pts_xyz.shape[0]
    pts_h = np.hstack([pts_xyz, np.ones((n, 1))])
    return (calib['velo_to_cam'] @ pts_h.T).T[:, :3]


def cam_to_image(pts_cam, calib):
    n = pts_cam.shape[0]
    pts_h = np.hstack([pts_cam, np.ones((n, 1))])
    img_h = (calib['cam_to_image'] @ pts_h.T).T
    img_h[:, :2] /= img_h[:, [2]]
    return img_h[:, :2]


def load_cam_points_with_rgb(frame_name, dataset_dir, calib):
    """Load velodyne scan, project to camera, filter to image FOV, append RGB."""
    import cv2
    bin_path  = os.path.join(dataset_dir, 'velodyne/training/velodyne', frame_name + '.bin')
    img_path  = os.path.join(dataset_dir, 'image/training/image_2',    frame_name + '.png')
    xyz_velo, intensity = load_velo_points(bin_path)
    xyz_cam = velo_to_cam(xyz_velo, calib)

    front_mask = xyz_cam[:, 2] > 0.1
    xyz_cam    = xyz_cam[front_mask]
    intensity  = intensity[front_mask]

    image = cv2.imread(img_path)                              # BGR uint8
    H, W  = image.shape[:2]                                   # actual image size

    img_pts = cam_to_image(xyz_cam, calib)
    in_img = ((img_pts[:, 0] > 0) & (img_pts[:, 0] < W) &
              (img_pts[:, 1] > 0) & (img_pts[:, 1] < H))
    xyz_cam   = xyz_cam[in_img]
    intensity = intensity[in_img]
    img_pts   = img_pts[in_img]

    u = np.clip(img_pts[:, 0].astype(int), 0, W - 1)
    v = np.clip(img_pts[:, 1].astype(int), 0, H - 1)
    bgr = image[v, u].astype(np.float32) / 255.0             # (N, 3) BGR [0,1]
    attr = np.hstack([intensity, bgr])                        # (N, 4): i, B, G, R
    return xyz_cam, attr


IGNORE_CLASSES = {'DontCare', 'Misc'}

def load_gt_labels(label_path):
    labels = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if fields[0] in IGNORE_CLASSES:
                continue
            labels.append({
                'name':   fields[0],
                'xmin':   float(fields[4]),
                'ymin':   float(fields[5]),
                'xmax':   float(fields[6]),
                'ymax':   float(fields[7]),
                'height': float(fields[8]),
                'width':  float(fields[9]),
                'length': float(fields[10]),
                'x3d':    float(fields[11]),
                'y3d':    float(fields[12]),
                'z3d':    float(fields[13]),
                'yaw':    float(fields[14]),
            })
    return labels


def load_pred_labels(pred_path):
    labels = []
    if not os.path.isfile(pred_path):
        return labels
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 16:
                continue
            labels.append({
                'name':   fields[0],
                'xmin':   float(fields[4]),
                'ymin':   float(fields[5]),
                'xmax':   float(fields[6]),
                'ymax':   float(fields[7]),
                'height': float(fields[8]),
                'width':  float(fields[9]),
                'length': float(fields[10]),
                'x3d':    float(fields[11]),
                'y3d':    float(fields[12]),
                'z3d':    float(fields[13]),
                'yaw':    float(fields[14]),
                'score':  float(fields[15]),
            })
    return labels


# ── 3D box geometry ────────────────────────────────────────────────────────────

def box_corners(x3d, y3d, z3d, l, h, w, yaw):
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                  [0,           1, 0           ],
                  [-np.sin(yaw),0, np.cos(yaw)]])
    c = np.array([[ l/2,  0,  w/2], [ l/2,  0, -w/2],
                  [-l/2,  0, -w/2], [-l/2,  0,  w/2],
                  [ l/2, -h,  w/2], [ l/2, -h, -w/2],
                  [-l/2, -h, -w/2], [-l/2, -h,  w/2]])
    return c.dot(R.T) + np.array([x3d, y3d, z3d])

BOX_EDGES = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]


# ── per-frame data loader ──────────────────────────────────────────────────────

def get_frame_data(frame_name):
    calib_path = os.path.join(DATASET_DIR, 'calib/training/calib', frame_name + '.txt')
    label_path = os.path.join(DATASET_DIR, 'labels/training/label_2', frame_name + '.txt')
    pred_path  = os.path.join(RESULTS_DIR, 'data', frame_name + '.txt')

    calib      = load_calib(calib_path)
    xyz, attr  = load_cam_points_with_rgb(frame_name, DATASET_DIR, calib)
    gt_labels  = load_gt_labels(label_path)
    pred_labels = [lb for lb in load_pred_labels(pred_path)
                   if lb['score'] >= SCORE_THRESH]
    return xyz, attr, gt_labels, pred_labels


# ── PLY export ─────────────────────────────────────────────────────────────────

def cam_to_cc(pts):
    """Remap KITTI camera coords to CloudCompare-friendly coords.
    Camera: X right, Y down, Z forward
    CloudCompare: X right, Y forward, Z up
    """
    return np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]])


def sample_box_edges(lb, color, n_pts=20):
    """Return densely sampled points along box edges so they look like lines."""
    crns = box_corners(lb['x3d'], lb['y3d'], lb['z3d'],
                       lb['length'], lb['height'], lb['width'], lb['yaw'])
    pts, colors = [], []
    for i, j in BOX_EDGES:
        for t in np.linspace(0, 1, n_pts):
            pts.append(crns[i] * (1 - t) + crns[j] * t)
            colors.append(color)
    return np.array(pts), np.array(colors)


def save_ply(frame_name):
    _, _, gt_labels, pred_labels = get_frame_data(frame_name)

    # Load velodyne points — filter to camera FOV only
    bin_path   = os.path.join(DATASET_DIR, 'velodyne/training/velodyne', frame_name + '.bin')
    calib_path = os.path.join(DATASET_DIR, 'calib/training/calib', frame_name + '.txt')
    img_path   = os.path.join(DATASET_DIR, 'image/training/image_2', frame_name + '.png')
    calib = load_calib(calib_path)
    xyz_velo, intensity = load_velo_points(bin_path)
    xyz_full = velo_to_cam(xyz_velo, calib)

    import cv2
    img = cv2.imread(img_path)
    H, W = img.shape[:2] if img is not None else (375, 1242)
    front_mask = xyz_full[:, 2] > 0.1                        # points in front of camera
    xyz_fov    = xyz_full[front_mask]
    intensity  = intensity[front_mask]
    img_pts    = cam_to_image(xyz_fov, calib)
    in_img     = ((img_pts[:, 0] >= 0) & (img_pts[:, 0] < W) &
                  (img_pts[:, 1] >= 0) & (img_pts[:, 1] < H))
    xyz_fov    = xyz_fov[in_img]
    intensity  = intensity[in_img]

    xyz_cc = cam_to_cc(xyz_fov)
    intensity_u8 = np.clip(intensity * 255, 0, 255).astype(np.uint8)
    rgb = np.repeat(intensity_u8, 3, axis=1)         # grey: R=G=B=intensity

    box_pts, box_rgb = [], []

    if args.show_gt:
        for lb in gt_labels:
            pts, cols = sample_box_edges(lb, GT_COLOR)
            box_pts.append(cam_to_cc(pts))
            box_rgb.append((cols * 255).astype(np.uint8))

    if args.show_pred:
        for lb in pred_labels:
            color = CLASS_COLORS.get(lb['name'], DEFAULT_PRED_COLOR)
            pts, cols = sample_box_edges(lb, color)
            box_pts.append(cam_to_cc(pts))
            box_rgb.append((cols * 255).astype(np.uint8))

    all_xyz = np.vstack([xyz_cc] + box_pts) if box_pts else xyz_cc
    all_rgb = np.vstack([rgb]    + box_rgb) if box_rgb else rgb

    out_path = os.path.join(OUTPUT_DIR, frame_name + '.ply')
    with open(out_path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(all_xyz)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in zip(all_xyz, all_rgb):
            f.write(f'{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n')
    print(f'  Saved PLY  → {out_path}')


# ── HTML export ────────────────────────────────────────────────────────────────

def save_html(frame_name):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print('plotly not found. Run:  pip install plotly')
        return

    xyz, attr, gt_labels, pred_labels = get_frame_data(frame_name)

    rgb_f = attr[:, 1:4]                      # BGR float [0,1]
    colors_str = [f'rgb({int(b*255)},{int(g*255)},{int(r*255)})'
                  for r, g, b in rgb_f]

    # Remap camera coords (X right, Y down, Z forward) → display coords
    # display: X=right, Y=forward(depth), Z=up  so the scene looks natural
    def remap(pts):
        # pts: (N, 3) in camera coords [X, Y, Z]
        return pts[:, 0], pts[:, 2], -pts[:, 1]   # x, z, -y

    dx, dy, dz = remap(xyz)
    traces = [go.Scatter3d(
        x=dx, y=dy, z=dz,
        mode='markers',
        marker=dict(size=1, color=colors_str),
        name='Point cloud', hoverinfo='skip',
    )]

    def add_box_trace(lb, color, label_str):
        crns = box_corners(lb['x3d'], lb['y3d'], lb['z3d'],
                           lb['length'], lb['height'], lb['width'], lb['yaw'])
        xs, ys, zs = [], [], []
        for i, j in BOX_EDGES:
            bx, by, bz = remap(crns[[i, j]])
            xs += [bx[0], bx[1], None]
            ys += [by[0], by[1], None]
            zs += [bz[0], bz[1], None]
        hex_c = '#{:02x}{:02x}{:02x}'.format(
            int(color[0]*255), int(color[1]*255), int(color[2]*255))
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines',
            line=dict(color=hex_c, width=3),
            name=label_str, hoverinfo='name',
        ))

    if args.show_gt:
        for lb in gt_labels:
            add_box_trace(lb, GT_COLOR, f'GT: {lb["name"]}')
    if args.show_pred:
        for lb in pred_labels:
            add_box_trace(lb, CLASS_COLORS.get(lb['name'], DEFAULT_PRED_COLOR),
                          f'Pred: {lb["name"]} ({lb["score"]:.2f})')

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f'Point-GNN — frame {frame_name}',
        scene=dict(
            xaxis_title='X (left/right)',
            yaxis_title='Z (forward/depth)',
            zaxis_title='Y (up)',
            aspectmode='data',
            camera=dict(eye=dict(x=0, y=-2.5, z=1.5)),  # slightly above, looking forward
        ),
        legend=dict(itemsizing='constant'),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    out_path = os.path.join(OUTPUT_DIR, frame_name + '.html')
    fig.write_html(out_path)
    print(f'  Saved HTML → {out_path}')


# ── BEV PNG export ─────────────────────────────────────────────────────────────

def save_bev(frame_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    xyz, attr, gt_labels, pred_labels = get_frame_data(frame_name)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    rgb_f = attr[:, 1:4][:, ::-1].clip(0, 1)  # BGR → RGB
    ax.scatter(xyz[:, 0], xyz[:, 2], c=rgb_f, s=0.3, alpha=0.6)

    def draw_bev_box(lb, color, linestyle='-', lw=2, score=None):
        crns = box_corners(lb['x3d'], lb['y3d'], lb['z3d'],
                           lb['length'], lb['height'], lb['width'], lb['yaw'])
        top = crns[:4]
        ax.add_patch(plt.Polygon(top[:, [0, 2]], closed=True, fill=False,
                                 edgecolor=color, linestyle=linestyle, linewidth=lw))
        front_mid = (top[0] + top[1]) / 2
        center    = top.mean(axis=0)
        ax.annotate('', xy=(front_mid[0], front_mid[2]),
                    xytext=(center[0], center[2]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))
        if score is not None:
            ax.text(lb['x3d'], lb['z3d'], f'{score:.2f}',
                    color=color, fontsize=7, ha='center', va='center')

    if args.show_gt:
        for lb in gt_labels:
            draw_bev_box(lb, color='lime', linestyle='--', lw=1.5)
    if args.show_pred:
        for lb in pred_labels:
            draw_bev_box(lb, color=CLASS_COLORS.get(lb['name'], DEFAULT_PRED_COLOR),
                     lw=2, score=lb['score'])

    ax.legend(handles=[
        mpatches.Patch(edgecolor='lime',    facecolor='none', ls='--', label='GT'),
        mpatches.Patch(edgecolor='red',     facecolor='none', label='Car (pred)'),
        mpatches.Patch(edgecolor='#0078ff', facecolor='none', label='Pedestrian (pred)'),
        mpatches.Patch(edgecolor='orange',  facecolor='none', label='Cyclist (pred)'),
    ], loc='upper right', facecolor='#2a2a3e', edgecolor='white', labelcolor='white')

    ax.set_xlabel('X (left/right)', color='white')
    ax.set_ylabel('Z (depth / forward)', color='white')
    ax.tick_params(colors='white')
    ax.set_title(f'BEV — frame {frame_name}  (score ≥ {SCORE_THRESH})',
                 color='white', fontsize=13)
    ax.set_xlim(-20, 20)
    ax.set_ylim(0, 70)
    ax.invert_yaxis()

    out_path = os.path.join(OUTPUT_DIR, frame_name + '_bev.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved BEV  → {out_path}')


# ── Camera image export ────────────────────────────────────────────────────────

def save_image(frame_name):
    import cv2

    _, _, gt_labels, pred_labels = get_frame_data(frame_name)

    img_path = os.path.join(DATASET_DIR, 'image/training/image_2', frame_name + '.png')
    image    = cv2.imread(img_path)

    # Color map for classes (BGR for OpenCV)
    CLASS_COLORS_BGR = {
        'Car':        (0,   50, 255),   # red
        'Pedestrian': (255, 120,  0),   # blue
        'Cyclist':    (0,  165, 255),   # orange
    }
    GT_BGR   = (0, 255, 0)              # green

    H, W = image.shape[:2]

    # ── project LiDAR depth onto image (colored by distance) ──────────────────
    if args.show_lidar:
        calib_path = os.path.join(DATASET_DIR, 'calib/training/calib', frame_name + '.txt')
        calib      = load_calib(calib_path)
        bin_path   = os.path.join(DATASET_DIR, 'velodyne/training/velodyne', frame_name + '.bin')
        xyz_velo, _ = load_velo_points(bin_path)
        xyz_cam     = velo_to_cam(xyz_velo, calib)
        front_mask  = xyz_cam[:, 2] > 0.1
        xyz_cam     = xyz_cam[front_mask]
        img_pts     = cam_to_image(xyz_cam, calib)
        in_img      = ((img_pts[:, 0] > 0) & (img_pts[:, 0] < W) &
                       (img_pts[:, 1] > 0) & (img_pts[:, 1] < H))
        img_pts = img_pts[in_img][::3]
        depth   = xyz_cam[in_img, 2][::3]
        max_d, min_d = depth.max(), depth.min()
        for (u, v), d in zip(img_pts.astype(int), depth):
            t = (d - min_d) / (max_d - min_d + 1e-6)
            color = (int(255 * t), 50, int(255 * (1 - t)))
            cv2.circle(image, (u, v), 1, color, -1)

    # ── draw GT boxes (green, dashed look via two rectangles) ─────────────────
    if args.show_gt:
        for lb in gt_labels:
            x1, y1 = int(lb['xmin']), int(lb['ymin'])
            x2, y2 = int(lb['xmax']), int(lb['ymax'])
            cv2.rectangle(image, (x1, y1), (x2, y2), GT_BGR, 2)
            cv2.putText(image, lb['name'], (x1, max(y1 - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, GT_BGR, 1)

    # ── draw predicted boxes (colored by class) ────────────────────────────────
    if args.show_pred:
        for lb in pred_labels:
            x1, y1 = int(lb['xmin']), int(lb['ymin'])
            x2, y2 = int(lb['xmax']), int(lb['ymax'])
            color  = CLASS_COLORS_BGR.get(lb['name'], (0, 0, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f'{lb["name"]} {lb["score"]:.2f}',
                        (x1, min(y2 + 14, H - 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    out_path = os.path.join(OUTPUT_DIR, frame_name + '_image.png')
    cv2.imwrite(out_path, image)
    print(f'  Saved IMG  → {out_path}')


# ── main ───────────────────────────────────────────────────────────────────────

EXPORT_FN = {'ply': save_ply, 'html': save_html, 'bev': save_bev, 'image': save_image}

file_list = read_split_file(SPLIT_FILE)
export_fn = EXPORT_FN[args.mode]

frames = range(len(file_list)) if args.frame == -1 else [args.frame]
for idx in frames:
    frame_name = file_list[idx]
    print(f'\nFrame {idx} ({frame_name})')
    export_fn(frame_name)

print(f'\nDone. Files saved to: {OUTPUT_DIR}')
