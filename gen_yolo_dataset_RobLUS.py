import os
import glob
import shutil
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import yaml

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SOURCE_DIR = os.path.join(BASE_DIR, 'RobLUS')
DATASET_OUTPUT_DIR = os.path.join(BASE_DIR, 'RobLUS_yolo_dataset')
CLASSES = ['pleural_line', 'rib_shadow']
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# --- Function to find contours and convert to YOLO format ---
def mask_to_yolo_segmentation(mask_path, image_width, image_height):
    """
    Converts a binary mask to YOLO segmentation format.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_segments = []
    for contour in contours:
        if contour.size >= 6:  # Need at least 3 points for a polygon
            segment = contour.flatten().tolist()
            # Normalize coordinates
            normalized_segment = [
                f"{coord / image_width if i % 2 == 0 else coord / image_height:.6f}"
                for i, coord in enumerate(segment)
            ]
            yolo_segments.append(" ".join(normalized_segment))
    return yolo_segments

# --- Main script ---
def create_yolo_dataset():
    """
    Generates the YOLO segmentation dataset.
    """
    print("--- Starting dataset creation process ---")
    # 1. Create output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DATASET_OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_OUTPUT_DIR, 'labels', split), exist_ok=True)

    # 2. Find all image paths
    image_paths = sorted(glob.glob(os.path.join(DATA_SOURCE_DIR, '*', 'US_*.jpg')))
    print(f"Found {len(image_paths)} images.")
    if not image_paths:
        print("Warning: No images found. Please check the DATA_SOURCE_DIR and file structure.")
        return

    # 3. Split into training and validation sets
    train_paths, val_paths = train_test_split(
        image_paths, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
    )

    # 4. Process each split
    for split, paths in [('train', train_paths), ('val', val_paths)]:
        print(f"Processing {len(paths)} images for the {split} set...")
        for i, img_path in enumerate(paths):
            # Get image details
            subject_dir = os.path.basename(os.path.dirname(img_path))
            original_img_filename = os.path.basename(img_path)
            
            # Create a new unique filename
            new_img_filename = f"{subject_dir}_{original_img_filename}"
            
            img_num_str = ''.join(filter(str.isdigit, os.path.splitext(original_img_filename)[0]))

            if not img_num_str:
                print(f"Could not extract number from {original_img_filename}, skipping.")
                continue
            
            img_num = int(img_num_str)

            # Get image dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image {img_path}, skipping.")
                continue

            h, w, _ = img.shape

            # Copy image to destination with the new unique name
            shutil.copy(img_path, os.path.join(DATASET_OUTPUT_DIR, 'images', split, new_img_filename))

            # --- Generate label file ---
            label_filename = os.path.splitext(new_img_filename)[0] + '.txt'
            label_path = os.path.join(DATASET_OUTPUT_DIR, 'labels', split, label_filename)

            with open(label_path, 'w') as f:
                for class_id, class_name in enumerate(CLASSES):
                    mask_filename = f'mask_{img_num}.png'
                    
                    mask_path = os.path.join(
                        DATA_SOURCE_DIR,
                        subject_dir,
                        'mask',
                        class_name,
                        mask_filename
                    )

                    if os.path.exists(mask_path):
                        segments = mask_to_yolo_segmentation(mask_path, w, h)
                        for segment in segments:
                            f.write(f"{class_id} {segment}\n")
            
            if (i + 1) % 50 == 0:
                print(f"  ... processed {i+1}/{len(paths)} images")

    # 5. Create dataset.yaml file
    print("Creating dataset.yaml file...")
    yaml_data = {
        'path': os.path.abspath(DATASET_OUTPUT_DIR),
        'train': os.path.join('images', 'train'),
        'val': os.path.join('images', 'val'),
        'names': {i: name for i, name in enumerate(CLASSES)}
    }

    with open(os.path.join(DATASET_OUTPUT_DIR, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)

    print("\nDataset generation complete!")
    print(f"Dataset created at: {os.path.abspath(DATASET_OUTPUT_DIR)}")

if __name__ == '__main__':
    # Check for dependencies
    try:
        import cv2
        import yaml
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"Error: Missing dependency - {e.name}")
        print("Please install the required packages by running:")
        print("pip install opencv-python-headless pyyaml scikit-learn")
        exit(1)
        
    create_yolo_dataset()