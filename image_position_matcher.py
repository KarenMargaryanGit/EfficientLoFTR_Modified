import copy
import os
import time
from copy import deepcopy

import numpy as np
import torch
import cv2 as cv
import re
import math
from typing import Tuple, List, Callable

from networkx import radius
from numpy.array_api import float32
from triton.language import dtype

from src.utils.plotting import make_matching_figures, make_matching_figure
from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
from util import *
import json

class ImagePositionMatcher:
    def __init__(self, folder: str, radius_km: float = 0.2):
        self.folder = folder
        json_path = os.path.join(self.folder, "tile_conf.json")
        with open(json_path, "r") as f:
            metadata = json.load(f)

        # Tile dimensions in pixels
        self.image_pixel_height = metadata["image_pixel_height"]
        self.image_pixel_width = metadata["image_pixel_width"]

        # Tile real-world dimensions in meters
        self.tile_height_m = metadata["image_meter_height"]
        self.tile_width_m = metadata["image_meter_width"]
        avg_lat = metadata["avg_latitude"]

        self.m_per_deg_lat = 111_320
        self.m_per_deg_lon = 111_320 * math.cos(math.radians(avg_lat))

        # Degrees per pixel
        self.deg_per_pixel_lat = (self.tile_height_m / self.image_pixel_height) / self.m_per_deg_lat
        self.deg_per_pixel_lon = (self.tile_width_m / self.image_pixel_width) / self.m_per_deg_lon

        # matcher setup
        self.model_type = 'full'
        self.precision = 'fp32'

        _default_cfg = deepcopy(opt_default_cfg)
        if self.model_type == 'full':
            _default_cfg = deepcopy(full_default_cfg)
        elif self.model_type == 'opt':
            _default_cfg = deepcopy(opt_default_cfg)

        if self.precision == 'mp':
            _default_cfg['mp'] = True
        elif self.precision == 'fp16':
            _default_cfg['half'] = True

        self.matcher =  LoFTR(config=_default_cfg)
        self.matcher.load_state_dict(torch.load("/home/garik/PycharmProjects/Eloftr/EfficientLoFTR/weights/eloftr_outdoor.ckpt",map_location=torch.device('cpu'))['state_dict'])
        self.matcher = reparameter(self.matcher)  # no reparameterization will lead to low performance

        if self.precision == 'fp16':
            self.matcher = self.matcher.half()

        self.matcher = self.matcher.eval()

        self.radius_km = radius_km
        self.index = self._index_images()

    def _index_images(self):
        index = []
        for filename in os.listdir(self.folder):
            if filename.endswith('.png'):
                try:
                    i, j, lat, lon = self._parse_filename(filename)
                    path = os.path.join(self.folder, filename)
                    index.append({'path': path, 'i': i, 'j': j, 'lat': lat, 'lon': lon})
                except ValueError:
                    continue
        return index

    def _parse_filename(self, filename: str) -> Tuple[int, int, float, float]:
        match = re.match(r"(\d+)_(\d+)_([-+]?[0-9]*\.?[0-9]+)_([-+]?[0-9]*\.?[0-9]+)\_cropped.png", filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        i, j, lat, lon = match.groups()
        return int(i), int(j), float(lat), float(lon)

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _find_candidates(self, lat: float, lon: float) -> List[str]:
        candidates = []
        for entry in self.index:
            dist = self._haversine(lat, lon, entry['lat'], entry['lon'])
            if dist <= self.radius_km:
                candidates.append(entry['path'])
        return candidates



    def match(self, frame, target_lat: float, target_lon: float, heading=0, method="cropped_4"):
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a NumPy array.")
        img0_r, rotation_matrix, rotated = self._rotate_if_needed(frame, heading)
        #print([[img0_r.shape[1], img0_r.shape[0]]])
        # cv.imshow("rotated", draw_keypoints(img0_r, [[img0_r.shape[1]/2, img0_r.shape[0]/2]], radius=5))
        # cv.waitKey(0)
        img0, scale_x, scale_y = self._resize_and_scale(img0_r)

        img0_tensor = self._to_tensor(img0)
        near_tiles, orientirs_gps = self._find_4_near_tiles(target_lat, target_lon)
        print(near_tiles)

        if method == "full_4":
            cropped_tile = self._merge_and_crop(near_tiles,  target_lat, target_lon, to_crop=False)
            img1, scale_x_2, scale_y_2 = self._resize_and_scale(cropped_tile)

            img1_tensor = self._to_tensor(img1)
        else:
            cropped_tile = self._merge_and_crop(near_tiles, target_lat, target_lon)
            scale_x_2, scale_y_2 = 1,1
            img1_tensor = self._to_tensor(cropped_tile)

        mkpts0, mkpts1, mconf = self._run_matcher(img0_tensor, img1_tensor)
        mkpts0_f, mkpts1_f = self._filter_matches(mkpts0, mkpts1, mconf,conf_thresh=0.3, use_ransac=True,   ransac_thresh=15)

        mkpts0_orig = mkpts0_f * [scale_x, scale_y]
        if rotated:
            mkpts0_orig = self.transform_pixels(mkpts0_orig, np.linalg.inv(rotation_matrix))
        mkpts1_orig = mkpts1_f * [scale_x_2, scale_y_2]

        print(mkpts0_f.shape, mkpts1_f.shape)
        matches = draw_matches(frame, mkpts0_orig, cropped_tile, mkpts1_orig)
        cv.imshow(f"matches with full", matches)
        cv.waitKey(0)
        # Optionally display
        if len(mkpts0_orig) >= 4 and len(mkpts1_orig) >= 4:
            H, mask = cv.findHomography(mkpts0_orig, mkpts1_orig, cv.RANSAC, 5.0)
            if H is not None:
                # Warp the original frame to align with the cropped_tile
                height, width = cropped_tile.shape[:2]
                status=check_homography(H, frame.shape[:2],cropped_tile.shape[:2])

                h, w = frame.shape[:2]
                corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32).reshape(-1, 1, 2)
                warped_corners = cv.perspectiveTransform(corners, H).reshape(-1, 2)


                center = np.array([w/2,h/2,1],dtype=np.float32)
                centroid = transformed = np.dot(H, center)
                centroid /= centroid[2]
                centroid = centroid[:2]
                warped_frame = cv.warpPerspective(frame, H, (width, height))

                # Optional: Show or return the warped image
                # cv.imshow(str(status), draw_keypoints(warped_frame, [centroid], radius=10))
                # cv.waitKey(0)
                print("orientir ", orientirs_gps)
                print("delta_y ", (centroid[1] - self.image_pixel_height/2))
                print("delta_x ", (centroid[0] - self.image_pixel_width/2))
                lat = orientirs_gps[0] - (centroid[1] - self.image_pixel_height/2)* self.deg_per_pixel_lat
                lon = orientirs_gps[1] + (centroid[0] - self.image_pixel_width/2)* self.deg_per_pixel_lon

                print(lat,lon)

        return mkpts0_orig, mkpts1_orig


    def _rotate_if_needed(self, frame, heading):
        if 3 <= heading <= 357:
            img_r, rot_mat = self.rotate_image(frame, -heading)
            return img_r, rot_mat, True
        return frame.copy(), np.eye(3), False

    def _resize_and_scale(self, img):
        H_orig, W_orig = img.shape[:2]

        # Target dimensions where the longer side is 640
        long_side = 640
        short_side = 448

        if H_orig >= W_orig:
            # Portrait or square orientation: height becomes 640
            H_new, W_new = long_side, short_side
        else:
            # Landscape orientation: width becomes 640
            W_new, H_new = long_side, short_side

        img_resized = cv.resize(img, (W_new, H_new))
        scale_x = W_orig / W_new
        scale_y = H_orig / H_new

        return img_resized, scale_x, scale_y

    def _to_tensor(self, img):
        tensor = torch.from_numpy(img)[None][None] / 255.
        if self.precision == 'fp16':
            tensor = tensor.half()
        return tensor

    def _run_matcher(self, img0, img1):
        batch = {'image0': img0, 'image1': img1}
        with torch.no_grad():
            start = time.time()
            if self.precision == 'mp':
                with torch.autocast(enabled=True, device_type='cpu'):
                    self.matcher(batch)
            else:
                self.matcher(batch)
            print("Inference time:", time.time() - start)
        return batch['mkpts0_f'].numpy(), batch['mkpts1_f'].numpy(), batch['mconf'].numpy()

    def _filter_matches(self, mkpts0, mkpts1, mconf, conf_thresh=0.3, use_ransac=True, ransac_thresh=3.0):
        if len(mkpts0) == 0:
            return None, None

        if self.model_type == 'opt':
            mconf = (mconf - min(20.0, mconf.min())) / (max(30.0, mconf.max()) - min(20.0, mconf.min()))

        # Step 1: Filter by confidence
        mask_conf = mconf > conf_thresh
        mkpts0_f = mkpts0[mask_conf]
        mkpts1_f = mkpts1[mask_conf]

        if len(mkpts0_f) < 4 or not use_ransac:
            # Not enough points for RANSAC or RANSAC disabled
            return mkpts0_f, mkpts1_f

        # Step 2: Filter with RANSAC (Homography)
        H, mask_ransac = cv.findHomography(mkpts0_f, mkpts1_f, cv.RANSAC, ransac_thresh)

        if mask_ransac is None:
            # RANSAC failed
            return np.empty((0, 2)), np.empty((0, 2)), mask_conf

        mask_ransac = mask_ransac.ravel().astype(bool)
        mkpts0_ransac = mkpts0_f[mask_ransac]
        mkpts1_ransac = mkpts1_f[mask_ransac]

        # Update full mask to reflect both filters
        #final_mask = mask_conf.copy()
        # Reset and re-apply only the RANSAC-inliers among the confident matches
        #final_mask[mask_conf] = mask_ransac

        return mkpts0_ransac, mkpts1_ransac  #final_mask



    def compute_warped_image_size(self,  H, image_shape):
        """
        Compute the size and pixel count of the image after homography warp.

        image_shape: (height, width) of the original image
        H: 3x3 homography matrix
        Returns: (width, height), pixel_count
        """
        h, w = image_shape[:2]

        # Define corners of the original image
        corners = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)

        # Convert to homogeneous coordinates
        corners_h = cv.perspectiveTransform(corners[None, :, :], H)[0]

        # Compute bounding box
        x_coords, y_coords = corners_h[:, 0], corners_h[:, 1]
        x_min, x_max = np.floor(x_coords.min()), np.ceil(x_coords.max())
        y_min, y_max = np.floor(y_coords.min()), np.ceil(y_coords.max())

        width = int(x_max - x_min)
        height = int(y_max - y_min)
        pixel_count = width * height

        return (width, height), pixel_count

    def _find_4_near_tiles(self, lat, lon) -> List[Tuple[int, int, str]]:
        """
        Return up to 4 surrounding tiles as (i, j, path)
        """
        # Find the closest tile center
        closest = min(self.index, key=lambda e: self._haversine(lat, lon, e['lat'], e['lon']))
        center_lat, center_lon = closest['lat'], closest['lon']
        dlat = lat - center_lat
        dlon = lon - center_lon

        # if position is within a quarter of tile's size → single tile is enough
        # if abs(dlat) < self.deg_per_pixel_lat * self.image_pixel_height / 10 and \
        #         abs(dlon) < self.deg_per_pixel_lon * self.image_pixel_width / 10:
        #     return [(closest['i'], closest['j'], closest['path'])]

        # else, find 4 surrounding tiles
        i0, j0 = closest['i'], closest['j']
        step_i = 1 if dlat < 0 else -1
        step_j = -1 if dlon < 0 else 1
        candidates = []
        for di in [0, step_i]:
            for dj in [0, step_j]:
                ii, jj = i0 + di, j0 + dj
                for entry in self.index:
                    if entry['i'] == ii and entry['j'] == jj:
                        candidates.append((ii, jj, entry['path']))

        for item in self.index:
            if item['i'] == i0 + min(step_i,0) and item['j'] == j0 + min(step_j,0):
                top_left_gps =  [item['lat'], item['lon']]

        return candidates, top_left_gps

    def _merge_and_crop(self, tiles: List[Tuple[int, int, str]], center_lat, center_lon, to_crop = True):

        h, w = self.image_pixel_height, self.image_pixel_width

        base_i, base_j = min((i, j) for i, j, _ in tiles)
        max_i = max(i for i, j, _ in tiles)
        max_j = max(j for i, j, _ in tiles)
        rows = max_i - base_i + 1
        cols = max_j - base_j + 1

        full = np.zeros((rows * h, cols * w), dtype=np.uint8)

        for i, j, path in tiles:
            row = i - base_i
            col = j - base_j
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            full[row * h:(row + 1) * h, col * w:(col + 1) * w] = img

        if not to_crop:
            return full
        # Use top-left tile as reference point for coordinate mapping
        ref_entry = next((e for e in self.index if e['i'] == base_i and e['j'] == base_j), None)
        if ref_entry is None:
            return None

        lat_ref = ref_entry['lat'] + (self.tile_height_m / 2) / self.m_per_deg_lat
        lon_ref = ref_entry['lon'] - (self.tile_width_m / 2) / self.m_per_deg_lon


        delta_lat = lat_ref - center_lat  # y-axis: lat decreases down
        delta_lon = center_lon - lon_ref  # x-axis: lon increases right

        px_y = int(delta_lat / self.deg_per_pixel_lat)
        px_x = int(delta_lon / self.deg_per_pixel_lon)

        # Shifted center relative to merged image
        center_y = px_y
        center_x = px_x

        rotate_and_crop_largest(full, 10)


        # Ensure the crop stays inside the full image
        center_y = max(h // 2, min(full.shape[0] - h // 2, center_y))
        center_x = max(w // 2, min(full.shape[1] - w // 2, center_x))


        # cv.imshow("full", draw_keypoints(full, [[center_x, center_y]], radius=5))
        # cv.waitKey(0)
        # Final crop bounds
        y1 = center_y - h // 2
        y2 = center_y + h // 2
        x1 = center_x - w // 2
        x2 = center_x + w // 2
        # Draw the rectangle (use tuples, not lists)
        cv.rectangle(full, (x1, y1), (x2, y2), (0, 255, 0), 10)

        # Display the image
        # cv.imshow("cropped", draw_keypoints(full, [[center_x, center_y]],radius=5))
        # cv.waitKey(0)
        crop = full[y1:y2, x1:x2]
        return crop

    def rotate_image(self,image, rotation_angle):
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        rotation_matrix = cv.getRotationMatrix2D(center, rotation_angle, 1.0)

        # Compute bounding box of the rotated image
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        rotated_corners = cv.transform(np.array([corners]), rotation_matrix)[0]
        min_x, min_y = np.min(rotated_corners, axis=0)
        max_x, max_y = np.max(rotated_corners, axis=0)

        # Adjust for translation
        translation_matrix = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ])

        homography_matrix = np.dot(translation_matrix, np.vstack([rotation_matrix, [0, 0, 1]]))

        rotated_image = cv.warpPerspective(image, homography_matrix, (
            int(max_x - min_x),
            int(max_y - min_y)
         ))
        # cv.imshow("", rotated_image)
        # cv.waitKey(2000)
        return rotated_image, homography_matrix

    import numpy as np

    def transform_pixels(self, points, matrix):
        def transform_point(point):
            point_homogeneous = np.array([point[0], point[1], 1])
            transformed = np.dot(matrix, point_homogeneous)
            return (transformed[:2] / transformed[2]).astype(int)

        # Handle single point (list or 1D array with 2 elements)
        if isinstance(points, (list, np.ndarray)) and np.array(points).ndim == 1 and len(points) == 2:
            return transform_point(points)

        # Handle multiple points (2D array)
        return np.array([transform_point(point) for point in points])


posMatcher = ImagePositionMatcher("cropped")
logs = []

with open(os.path.join("Flight_4", "logs_14_10.json"), 'r') as file:
    for line in file:
        logs.append(json.loads(line))


for i in range(2,9,1):
    target_lat, target_lon = logs[i]["vslam_gps"]
    heading = (logs[i]["heading"] + 270)%360
    print(target_lat, target_lon)
    posMatcher.match(cv.imread(f"Flight_4/image_{i}_original.jpg", cv.IMREAD_GRAYSCALE), target_lat, target_lon, heading = heading, method = "full_4")



# with open(os.path.join("../../../logger/Flight_4", "logs.json"), 'r') as file:
#     for line in file:
#         logs.append(json.loads(line))
#
# for i in range(300,3000,50):
#     target_lat, target_lon = logs[i]["gps"]
#     heading = (logs[i]["heading"] + 270)%360
#     #print(target_lat, target_lon)
#     posMatcher.match(cv.imread(f"../../../logger/Flight_4/image_{i}.jpg", cv.IMREAD_GRAYSCALE), target_lat, target_lon, heading = heading, method= "full_4")
