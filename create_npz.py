
import numpy as np
import os

# --- 1. Define your data paths ---
dataset_path = 'data/my_dataset'
image_dir = os.path.join(dataset_path, 'images')
depth_dir = os.path.join(dataset_path, 'depth')
output_npz_path = os.path.join(dataset_path, 'scene_info_files', 'scene2.npz')

# --- 2. Define your image pairs ---
# IMPORTANT: You must manually define your image pairs here.
# The format is [('image1_name.png', 'image2_name.png'), ...]
pair_list = [
    ('pair_0002/drone.png', 'pair_0002/satellite.png'),
    # Add your image pairs here, for example:
    # ('1_1_40.22596592075515_43.9188770002409_cropped.png', '1_2_40.22596592075515_43.92158615719963_cropped.png'),
    # ('2_1_40.22447304688932_43.9188770002409_cropped.png', '2_2_40.22447304688932_43.92158615719963_cropped.png'),
]

# --- 3. Get all unique images and create mappings ---
all_images = sorted(list(set([img for pair in pair_list for img in pair])))
image_to_index = {img: i for i, img in enumerate(all_images)}

image_paths = [os.path.join('images', f) for f in all_images]
depth_paths = [os.path.join('depth', f) for f in all_images] # Assuming depth has the same name

num_images = len(all_images)

# --- 4. Define Intrinsics and Poses (CRITICAL) ---
# YOU MUST REPLACE THESE PLACEHOLDERS WITH YOUR REAL DATA

# Intrinsics: A list of 3x3 numpy arrays, one for each image.
# This is the camera matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
intrinsics = []
for img_name in all_images:
    # Example for a 640x448 image. Replace with your camera's focal length and principal point.
    fx, fy = 525.0, 525.0 # Example focal lengths
    cx, cy = 319.5, 223.5 # Example principal point
    intrinsics.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))

# Poses: A list of 4x4 numpy arrays (camera-to-world transformation), one for each image.
# If all your images are from a static camera, you can use identity matrices.
# If the camera moves, you MUST provide the correct pose for each image.
poses = [np.eye(4) for _ in range(num_images)]

# --- 5. Create the pair_infos array ---
pair_infos = []
for img1_name, img2_name in pair_list:
    idx0 = image_to_index[img1_name]
    idx1 = image_to_index[img2_name]
    pair_infos.append(
        (
            (idx0, idx1),      # Indices of the image pair
            0.8,             # A default overlap score (you can refine this)
            np.array([])     # Placeholder for central_matches
        )
    )

# --- 6. Save to .npz file ---
np.savez(
    output_npz_path,
    image_paths=np.array(image_paths, dtype=object),
    depth_paths=np.array(depth_paths, dtype=object),
    intrinsics=np.array(intrinsics),
    poses=np.array(poses),
    pair_infos=np.array(pair_infos, dtype=object)
)

print(f"Successfully created {output_npz_path}")
