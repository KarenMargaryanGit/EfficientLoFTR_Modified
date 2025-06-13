import copy
import os
import logging
import time

import torch
import cv2 as cv
from pythonjsonlogger import jsonlogger
import numpy as np
import math


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""

    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def load_features(file_path):
    print(file_path)
    data = torch.load(file_path)
    keypoints = data['keypoints']
    descriptors = data['descriptors']
    scores = data['scores']
    image_size = data['base_image_size']
    #pile_size = data['pile_image_size']
    base_feats = {'keypoints': keypoints, 'keypoint_scores': scores, 'descriptors': descriptors, 'image_size': image_size}

    return base_feats


def nearest_point_from_center(coords, width, height):
    coords = np.asarray(coords)
    assert coords.shape[1] == 2, "Coordinates should be of shape (N, 2)"
    center = np.array([[width / 2, height / 2]])
    distances = np.linalg.norm(coords - center, axis=1)
    return np.argmin(distances)


def keypoints_in_circle(query_points, radius, center):
    dist = np.linalg.norm(query_points - center, axis=1)
    idx = dist <= radius
    return idx


def point_estimation_with_homography(query_points, base_points, center):
    if len(query_points) > 3:
        H, status = cv.findHomography(query_points, base_points)
        transformed = np.dot(H, center)
        transformed /= transformed[2]
        centroid = np.array([transformed[0], transformed[1]])
        if centroid[0] < 0 or centroid[1] < 0:
            return None, None
        return centroid, H
    else:
        return None, None


def weighted_mean(coords, scores, threshold=0.0):
    # Ensure coords is a numpy array of shape (N, 2)
    coords = np.asarray(coords)
    assert coords.shape[1] == 2, "Coordinates should be of shape (N, 2)"

    # Ensure scores is a numpy array of shape (N,)
    scores = np.asarray(scores)
    assert scores.shape[0] == coords.shape[0], "Scores should be of shape (N,)"

    # Filter points based on the threshold
    valid_indices = scores > threshold
    filtered_coords = coords[valid_indices]
    filtered_scores = scores[valid_indices]

    # Check if there are any valid points
    if len(filtered_coords) == 0:
        raise ValueError("No points have scores above the given threshold.")

    # Compute the weighted sum of coordinates
    weighted_sum = np.sum(filtered_coords * filtered_scores[:, np.newaxis], axis=0)

    # Compute the sum of the weights
    sum_of_weights = np.sum(filtered_scores)

    # Compute the weighted mean
    weighted_mean_coords = weighted_sum / sum_of_weights

    return weighted_mean_coords


def rotation_angle(drone_angle, camera_angle=0, scraped_angle=0, rotated_is_available=False):
    relative_angle = (drone_angle + camera_angle - scraped_angle) % 360


    use_rotated = rotated_is_available and (
        22.5 <= relative_angle < 67.5 or
        112.5 <= relative_angle < 157.5 or
        202.5 <= relative_angle < 247.5 or
        292.5 <= relative_angle < 337.5
    )
    if rotated_is_available:
        if 67.5 <= relative_angle < 157.5:
            rotation = cv.ROTATE_90_CLOCKWISE
        elif 157.5 <= relative_angle < 247.5:
            rotation = cv.ROTATE_180
        elif 247.5 <= relative_angle < 337.5:
            rotation = cv.ROTATE_90_COUNTERCLOCKWISE
        else:
            rotation = None
    else:
        if 45 <= relative_angle < 135:
            rotation = cv.ROTATE_90_CLOCKWISE
        elif 135 <= relative_angle < 225:
            rotation = cv.ROTATE_180
        elif 225 <= relative_angle < 315:
            rotation = cv.ROTATE_90_COUNTERCLOCKWISE
        else:
            rotation = None

    return use_rotated, rotation



def point_to_line_distance(A, B, P):
    """
    Computes the perpendicular distance from point P to the line through points A and B.

    Parameters:
    A, B, P: Tuples or lists representing the (x, y) coordinates of the points.

    Returns:
    Distance from point P to the line AB.
    """
    # Unpack the coordinates
    x1, y1 = A
    x2, y2 = B
    x3, y3 = P

    # Calculate the numerator of the distance formula
    numerator = abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1)

    # Calculate the denominator of the distance formula
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    # Return the distance
    return numerator / denominator


def distance_l2(point1, point2):
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    return math.sqrt(delta_x ** 2 + delta_y ** 2)


def compute_velocity(drone_pose, target_pose, speed):
    delta_x = (target_pose[0] - drone_pose[0]) / 4
    delta_y = (target_pose[1] - drone_pose[1]) / 4

    # Calculate the distance and duration
    distance = math.sqrt(delta_x ** 2 + delta_y ** 2)

    duration = distance / speed

    # Calculate the velocities
    velocity_y = delta_x / duration
    velocity_x = -delta_y / duration
    return  [velocity_x, velocity_y], 4


def draw_keypoints(image, keypoints, radius = 1, color=(0, 255, 0)):
    image_copy = copy.deepcopy(image)
    for point in keypoints:
        cv.circle(image_copy, (int(point[0]), int(point[1])), radius=radius, color = color, thickness=-1)
    return image_copy


def draw_matches(query_img, query_kps, base_img, base_kps):
    if len(query_kps) == 0 and len(base_kps) == 0:
        print("No matches found.")
        return
    keypoints1 = [cv.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in query_kps]
    keypoints2 = [cv.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in base_kps]


    matches_lightglue = [(i, i) for i in range(base_kps.shape[0])]  # Matches from LightGlue (e.g., (i, j) where i matches j)

    # Convert the LightGlue matches into cv2.DMatch objects
    matches = []
    for i, j in matches_lightglue:
        match = cv.DMatch(_queryIdx=i, _trainIdx=j, _imgIdx=0, _distance=0)  # Set distance to 0 or some computed value
        matches.append(match)

    return cv.drawMatches(query_img, keypoints1, base_img, keypoints2, matches, None)


def resize_to_fit(image, width=1920, height = 1080):
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)


def create_logger(logger_path):
    counter = 0
    while os.path.exists(logger_path + "Flight_" + str(counter)):
        counter += 1

    logger_path += f"Flight_{counter}/"
    os.makedirs(logger_path)

    logger = logging.getLogger("vslam_logger")
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    file_handler = logging.FileHandler(logger_path + "logs.json", mode='w')
    file_handler.setLevel(logging.DEBUG)

    # Create a JSON formatter
    json_formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s'
    )
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    return logger, logger_path


def rotate_pixel(rotation_code, pixel, image_shape=(720, 960)):
    h, w = image_shape
    x, y = pixel

    if rotation_code == cv.ROTATE_90_COUNTERCLOCKWISE:
        new_x = y
        new_y = w - 1 - x
    elif rotation_code == cv.ROTATE_180:
        new_x = w - 1 - x
        new_y = h - 1 - y
    elif rotation_code == cv.ROTATE_90_CLOCKWISE:
        new_x = h - 1 - y
        new_y = x
    else:
        new_x, new_y = x, y

    return [new_x, new_y]

def draw_filtered_keypoints(base_img_path, filtered_keypoints, center):
    base_image = cv.imread(base_img_path)
    for point in filtered_keypoints[0]:
        point = point.tolist()
        cv.circle(base_image, (int(point[0]), int(point[1])), radius=20, color=(0, 255, 0), thickness=-1)

    cv.circle(base_image, (int(center[0]), int(center[1])), radius=20, color=(255, 255, 0),
              thickness=-1)
    base_image = resize_to_fit(base_image, width=1300, height=700)

    cv.destroyAllWindows()
    cv.imshow("filtered_keypoints", base_image)
    cv.waitKey(0)

def haversine(lat1, lon1, lat2, lon2):
    """Computes the great-circle distance between two GPS points using the Haversine formula."""
    R = 6371000  # Earth's radius in meters

    # Convert degrees to radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def check_homography(H, src_shape, dst_shape):
    h, w = src_shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv.perspectiveTransform(corners, H).reshape(-1, 2)
    #print(warped_corners)
    area = quadrilateral_area(warped_corners)
    is_convex = is_convex_quad(warped_corners)
    scale_ratio = homography_scale_check(H)
    det = np.linalg.det(H[:2, :2])

    #print(f"Area: {area:.2f}, Convex: {is_convex}, Scale Ratio: {scale_ratio:.2f}, Det: {det:.2f}")

    # Check image bounds
    x_min, y_min = warped_corners.min(axis=0)
    x_max, y_max = warped_corners.max(axis=0)
    fits_in_dest = (x_max - x_min <= dst_shape[1]) and (y_max - y_min <= dst_shape[0])
    #print("fits_in_dest: ", fits_in_dest)
    return {
        "area": area,
        "convex": is_convex,
        "scale_ratio": scale_ratio,
        "det": det,
        "fits_in_dest": fits_in_dest
    }
def quadrilateral_area(pts):
    pts = np.array(pts)
    return 0.5 * np.abs(np.dot(pts[:,0], np.roll(pts[:,1], 1)) - np.dot(pts[:,1], np.roll(pts[:,0], 1)))

def homography_scale_check(H):
    A = H[:2, :2]
    u, s, vh = np.linalg.svd(A)
    scale_ratio = s.max() / s.min()
    return scale_ratio  # should ideally be < ~10


def is_convex_quad(pts):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    pts = np.array(pts)
    signs = []
    for i in range(4):
        o, a, b = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        signs.append(np.sign(cross(o, a, b)))

    return np.all(np.array(signs) > 0) or np.all(np.array(signs) < 0)

def line_intersection(p1, p2, q1, q2):
    """
    Finds the intersection point of two lines (p1-p2 and q1-q2).
    Each point is a tuple or array: (x, y)
    """
    # Convert points to numpy arrays
    p1, p2 = np.array(p1), np.array(p2)
    q1, q2 = np.array(q1), np.array(q2)

    # Line vectors
    dp = p2 - p1
    dq = q2 - q1

    # Solve for t in: p1 + t * dp = q1 + u * dq
    matrix = np.array([dp, -dq]).T
    if np.linalg.matrix_rank(matrix) < 2:
        raise ValueError("Lines are parallel or degenerate")

    rhs = q1 - p1
    t, _ = np.linalg.solve(matrix, rhs)

    intersection = p1 + t * dp
    return tuple(intersection)


import cv2
import numpy as np


def largest_rotated_rect(w, h, angle_rad):
    if w <= 0 or h <= 0:
        return 0, 0

    quadrant = int(np.floor(angle_rad / (np.pi / 2))) & 3
    sign_alpha = angle_rad if (quadrant & 1) == 0 else np.pi - angle_rad
    alpha = (sign_alpha % np.pi + np.pi) % np.pi

    bb_w = w * np.cos(alpha) + h * np.sin(alpha)
    bb_h = w * np.sin(alpha) + h * np.cos(alpha)

    gamma = np.arctan2(h, w) if w < h else np.arctan2(w, h)
    delta = np.pi - alpha - gamma

    length = h if w < h else w
    d = length * np.cos(alpha)
    a = d * np.sin(alpha) / np.sin(delta)

    y = a * np.cos(gamma)
    x = y * np.tan(gamma)

    return (int(bb_w - 2 * x), int(bb_h - 2 * y))


def rotate_and_crop_largest(img, angle_deg):
    h, w = img.shape[:2]
    angle_rad = np.deg2rad(angle_deg)
    center = (w / 2, h / 2)

    # Step 1: Compute rotation matrix
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # Step 2: Compute new image bounds
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the matrix for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Step 3: Rotate the image
    rotated = cv2.warpAffine(img, M, (new_w, new_h), borderValue=(0, 0, 0))

    # Step 4: Compute the largest rectangle without black borders
    crop_w, crop_h = largest_rotated_rect(w, h, angle_rad)

    # Step 5: Center crop
    x = (new_w - crop_w) // 2
    y = (new_h - crop_h) // 2
    cropped = rotated[y:y + crop_h, x:x + crop_w]
    # cv.imshow("rot,crop", cropped)
    # cv.waitKey(0)
    return cropped, M, (x, y)


