import cv2
import numpy as np

# Paths to your images
img0_path = 'train_dataset_sample_copy/images/pair_0002/drone.png'
img1_path = 'train_dataset_sample_copy/images/pair_0002/satellite.png'

# Camera intrinsics (example, replace with your calibration)
K = np.array([
    [1500,    0, 960],
    [   0, 1500, 540],
    [   0,    0,   1]
], dtype=np.float32)

# Read images
img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

# Compute SIFT keypoints and descriptors
sift = cv2.SIFT_create()
kp0, des0 = sift.detectAndCompute(img0, None)
kp1, des1 = sift.detectAndCompute(img1, None)

# Match descriptors
bf = cv2.BFMatcher()
matches = bf.knnMatch(des0, des1, k=2)

# Ratio test
good_matches = []
pts0 = []
pts1 = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        pts0.append(kp0[m.queryIdx].pt)
        pts1.append(kp1[m.trainIdx].pt)
pts0 = np.array(pts0)
pts1 = np.array(pts1)

# Use RANSAC to filter inliers
if len(pts0) >= 8:
    F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 3.0, 0.99)
    inlier_pts0 = pts0[mask.ravel() == 1]
    inlier_pts1 = pts1[mask.ravel() == 1]
else:
    inlier_pts0 = np.empty((0, 2))
    inlier_pts1 = np.empty((0, 2))

print(f"Original matches: {len(pts0)}, Inliers after RANSAC: {len(inlier_pts0)}")

# Save .npz file for EfficientLoFTR
np.savez('train_dataset_sample_copy/meta/pair_0002.npz',
         image0_path=img0_path,
         image1_path=img1_path,
         K0=K,
         K1=K,
         matches0=inlier_pts0,  # Nx2
         matches1=inlier_pts1   # Nx2
         )
print('Saved train_dataset_sample_copy/meta/pair_0001.npz')