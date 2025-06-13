import cv2
import os

# Configuration
input_folder = "zovuni"           # Your input folder
output_folder = "zovuni"    # Set to input_folder to overwrite
new_size = (640, 448)             # (width, height)

# Create output folder if needed
os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# Process each image
for filename in os.listdir(input_folder):
    if filename.lower().endswith(extensions):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Read image
        img = cv2.imread(input_path)

        if img is None:
            print(f"Skipping invalid image: {filename}")
            continue

        # Resize
        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Save
        cv2.imwrite(output_path, gray)
        print(f"Processed: {filename}")
