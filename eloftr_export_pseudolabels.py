"""
Run ELoFTR matching and export pseudo-labels for student model training.

For each (img0, img1) matching pair, saves a .npz file with:
    - indexB: (HW,) array of matched indices in B for each grid point in A
    - mask: (HW,) array, 1 if match is valid, 0 if not

You must run this on all your training pairs to generate labels for training your student model.
"""

import os
import numpy as np
import torch
import cv2 as cv
import json
from copy import deepcopy
from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

from image_position_matcher import ImagePositionMatcher  # Assuming this is your custom matcher module
import os
import numpy as np
import torch
import cv2 as cv
import json
from copy import deepcopy
from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

def grid_indices_from_points(points, H, W, downscale=8):
    y = np.clip((points[:, 1] / downscale).astype(int), 0, H // downscale - 1)
    x = np.clip((points[:, 0] / downscale).astype(int), 0, W // downscale - 1)
    return y * (W // downscale) + x

def save_pseudolabel(img0_path, img1_path, mkpts0, mkpts1, mconf, out_dir, input_size=(320,320), grid_size=8, conf_thresh=0.3):
    H, W = input_size
    HW = (H // grid_size) * (W // grid_size)
    indexB = np.full(HW, -1, dtype=np.int64)
    mask = np.zeros(HW, dtype=np.float32)

    keep = mconf > conf_thresh
    mkpts0 = mkpts0[keep]
    mkpts1 = mkpts1[keep]

    if len(mkpts0) == 0 or len(mkpts1) == 0:
        print(f"WARNING: No matches above threshold for {img0_path} <-> {img1_path}")
        baseA = os.path.splitext(os.path.basename(img0_path))[0]
        baseB = os.path.splitext(os.path.basename(img1_path))[0]
        np.savez_compressed(
            os.path.join(out_dir, f"{baseA}_{baseB}.npz"),
            indexB=indexB,
            mask=mask
        )
        return

    idxA = grid_indices_from_points(mkpts0, H, W, downscale=grid_size)
    idxB = grid_indices_from_points(mkpts1, H, W, downscale=grid_size)

    for a, b in zip(idxA, idxB):
        indexB[a] = b
        mask[a] = 1.0

    baseA = os.path.splitext(os.path.basename(img0_path))[0]
    baseB = os.path.splitext(os.path.basename(img1_path))[0]
    np.savez_compressed(
        os.path.join(out_dir, f"{baseA}_{baseB}.npz"),
        indexB=indexB,
        mask=mask
    )

def prep(img, input_size=(320,320)):
    img = cv.resize(img, input_size)
    img = torch.from_numpy(img)[None][None] / 255.
    return img.float()

def run_and_export(folder, logs_path, out_label_dir, input_size=(320,320), grid_size=8, conf_thresh=0.3):
    model_type = 'full'
    precision = 'fp32'
    _default_cfg = deepcopy(opt_default_cfg)
    if model_type == 'full':
        _default_cfg = deepcopy(full_default_cfg)
    elif model_type == 'opt':
        _default_cfg = deepcopy(opt_default_cfg)
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("weights/eloftr_outdoor.ckpt", map_location=torch.device('cpu'))['state_dict'])
    matcher = reparameter(matcher)
    matcher = matcher.eval()

    with open(logs_path, "r") as f:
        logs = [json.loads(line) for line in f if line.strip()]

    os.makedirs(out_label_dir, exist_ok=True)
    posMatcher = ImagePositionMatcher(folder)

    for entry in logs:
        if entry.get("vslam_gps") is None:
            continue
        i = entry["id"]
        target_lat, target_lon = entry["vslam_gps"]
        heading = (entry["heading"] + 270) % 360

        img0_path = os.path.join(folder, f"Flight_4/image_{i}_original.jpg")
        if not os.path.exists(img0_path):
            print(f"WARNING: {img0_path} missing, skipping.")
            continue
        img0 = cv.imread(img0_path, cv.IMREAD_GRAYSCALE)
        if img0 is None:
            print(f"ERROR: Could not load img0 at {img0_path}")
            continue

        near_tiles, orientirs_gps = posMatcher._find_4_near_tiles(target_lat, target_lon)
        cropped_tile = posMatcher._merge_and_crop(near_tiles, target_lat, target_lon)
        if cropped_tile is None:
            print(f"ERROR: Could not generate cropped_tile for {img0_path}")
            continue
        img1 = cropped_tile
        img1_path = os.path.join(folder, f"temp_cropped_{i}.png")
        cv.imwrite(img1_path, img1)

        img0_tensor = prep(img0, input_size=input_size)
        img1_tensor = prep(img1, input_size=input_size)

        batch = {'image0': img0_tensor, 'image1': img1_tensor}
        with torch.no_grad():
            matcher(batch)
        mkpts0 = batch['mkpts0_f'].numpy()
        mkpts1 = batch['mkpts1_f'].numpy()
        mconf = batch['mconf'].numpy()

        save_pseudolabel(img0_path, img1_path, mkpts0, mkpts1, mconf, out_label_dir, input_size=input_size, grid_size=grid_size, conf_thresh=conf_thresh)
        print(f"Saved pseudo-label for {os.path.basename(img0_path)} <-> {os.path.basename(img1_path)}")

if __name__ == "__main__":
    run_and_export(
        folder="cropped",
        logs_path=os.path.join("Flight_4", "logs_14_10.json"),
        out_label_dir="data/pseudo_labels",
        input_size=(320, 320),
        grid_size=8,
        conf_thresh=0.3
    )