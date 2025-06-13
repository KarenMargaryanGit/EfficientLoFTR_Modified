import os
import csv
import cv2
import numpy as np
import torch
from pyproj import Transformer

# --------------------------------------------
# Utilities for geodetic to ECEF/ENU conversion
# --------------------------------------------
transformer_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

def geodetic_to_ecef(lat, lon, alt):
    x, y, z = transformer_ecef.transform(lon, lat, alt)
    return np.array([x, y, z])

class ENUConverter:
    def __init__(self, ref_lat, ref_lon, ref_alt):
        self.ref_lat = np.deg2rad(ref_lat)
        self.ref_lon = np.deg2rad(ref_lon)
        self.ref_ecef = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)

    def geodetic_to_enu(self, lat, lon, alt):
        p_ecef = geodetic_to_ecef(lat, lon, alt)
        d = p_ecef - self.ref_ecef
        sin_lat, cos_lat = np.sin(self.ref_lat), np.cos(self.ref_lat)
        sin_lon, cos_lon = np.sin(self.ref_lon), np.cos(self.ref_lon)
        e = -sin_lon * d[0] + cos_lon * d[1]
        n = -sin_lat * cos_lon * d[0] - sin_lat * sin_lon * d[1] + cos_lat * d[2]
        u =  cos_lat * cos_lon * d[0] + cos_lat * sin_lon * d[1] + sin_lat * d[2]
        return np.array([e, n, u])

# --------------------------------------------
# Depth estimation via MiDaS DPT models
# --------------------------------------------

def load_depth_model(model_type='DPT_Large', device='cpu'):
    key = model_type.strip().lower()
    if key == 'dpt_large': hub_name = 'DPT_Large'
    elif key == 'dpt_hybrid': hub_name = 'DPT_Hybrid'
    else: raise ValueError(f"Unsupported depth model type: {model_type}")
    model = torch.hub.load('intel-isl/MiDaS', hub_name)
    model.to(device).eval()
    transform = torch.hub.load('intel-isl/MiDaS', 'transforms').dpt_transform
    return model, transform


def generate_depth_map(img_rgb, model, transform, device='cpu', scale_factor=100.0):
    with torch.no_grad():
        input_batch = transform(img_rgb).to(device)
        pred = model(input_batch).unsqueeze(1)
        pred = torch.nn.functional.interpolate(pred, size=img_rgb.shape[:2], mode='bicubic', align_corners=False)
        depth = pred.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    return (depth * scale_factor).astype(np.float32)

# --------------------------------------------
# Core dataset preparation
# --------------------------------------------

def load_metadata(csv_path):
    pairs = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)
    return pairs


def compute_extrinsics(pair, enu_converter):
    pA = enu_converter.geodetic_to_enu(float(pair['lat_a']), float(pair['lon_a']), float(pair['alt_a']))
    pB = enu_converter.geodetic_to_enu(float(pair['lat_b']), float(pair['lon_b']), float(pair['alt_b']))
    return np.eye(3), pB - pA


def warp_grid(shape, intrinsics, R, t, depth_map, grid_size=16):
    H, W = shape[:2]
    xs = np.linspace(0, W-1, grid_size).astype(int)
    ys = np.linspace(0, H-1, grid_size).astype(int)
    K, Kinv = intrinsics['K'], np.linalg.inv(intrinsics['K'])
    coordsA, coordsB = [], []
    for y in ys:
        for x in xs:
            Z = depth_map[y, x]
            uv = np.array([x, y, 1.], dtype=np.float32)
            X = Z * (Kinv @ uv)
            Xb = R @ X + t
            if Xb[2] <= 0:
                continue
            proj = K @ (Xb / Xb[2])
            u2, v2 = int(proj[0]), int(proj[1])
            if 0 <= u2 < W and 0 <= v2 < H:
                coordsA.append((x, y))
                coordsB.append((u2, v2))
    return coordsA, coordsB


def process_dataset(csv_path, output_dir, intrinsics,
                    model_type='DPT_Large', device='cpu', scale_factor=100.0,
                    ground_alt=0.0, grid_size=16, use_grayscale=False):
    os.makedirs(output_dir, exist_ok=True)
    depth_dir = os.path.join(output_dir, 'depth_maps')
    os.makedirs(depth_dir, exist_ok=True)

    pairs = load_metadata(csv_path)
    if not pairs:
        print("No pairs in CSV.")
        return

    # Load depth model
    depth_model, depth_transform = load_depth_model(model_type, device)
    # Setup ENU converter with first pair
    ref = pairs[0]
    enu_conv = ENUConverter(float(ref['lat_a']), float(ref['lon_a']), float(ref['alt_a']))

    for i, pair in enumerate(pairs):
        # Read paths and rotation angle from metadata
        Apath = pair['image_a_path'].strip()
        Bpath = pair['image_b_path'].strip()
        angle = float(pair.get('angle', 0.0))

        # Load images
        A = cv2.imread(Apath, cv2.IMREAD_UNCHANGED)
        B = cv2.imread(Bpath, cv2.IMREAD_UNCHANGED) if Bpath else None
        if A is None:
            print(f"Pair {i}: cannot load A={Apath}")
            continue

        # Apply rotation to A only
        if angle != 0.0:
            hA, wA = A.shape[:2]
            centerA = (wA/2, hA/2)
            M = cv2.getRotationMatrix2D(centerA, angle, 1.0)
            A = cv2.warpAffine(A, M, (wA, hA), borderValue=(0,0,0))

        # Optional grayscale conversion
        if use_grayscale:
            gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY) if A.ndim > 2 else A
            A = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Split grid if needed
        if B is None:
            h, w = A.shape[:2]
            mid = w // 2
            B = A[:, mid:]
            A = A[:, :mid]
            print(f"Pair {i}: split grid at x={mid}")

        # Prepare RGB for depth
        A_rgb = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)

        # Compute extrinsics
        R, t = compute_extrinsics(pair, enu_conv)
        print(f"Pair {i}: rotation angle={angle}, translation t={t}")

        # Generate and save depth map for A
        depthA = generate_depth_map(A_rgb, depth_model, depth_transform, device, scale_factor)
        raw_file = os.path.join(depth_dir, f"pair_{i:03d}_depthA.npy")
        np.save(raw_file, depthA)
        vis = cv2.normalize(depthA, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(depth_dir, f"pair_{i:03d}_depthA.png"), vis)

        # Warp coarse grid to get matches
        coordsA, coordsB = warp_grid(A.shape, intrinsics, R, t, depthA, grid_size)
        if not coordsA:
            print(f"Pair {i}: no matches")
            continue

        # Save matches
        out_file = os.path.join(output_dir, f"pair_{i:03d}_matches.npz")
        np.savez_compressed(out_file, coordsA=np.array(coordsA), coordsB=np.array(coordsB))
        print(f"Pair {i}: {len(coordsA)} matches saved")

if __name__ == '__main__':
    CSV = 'data/drone_pairs.csv'
    OUT = 'data/prepared_pairs'
    K = np.array([[1200, 0, 640], [0, 1200, 480], [0, 0, 1]], dtype=np.float32)
    process_dataset(CSV, OUT, {'K': K}, device='cpu', use_grayscale=True)
