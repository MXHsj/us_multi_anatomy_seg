import os
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt')

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USE_TWO_DATASETS = True  # Set to True to combine both datasets for training

if USE_TWO_DATASETS:
    # Combine cae_lung_phantom and RobLUS datasets
    dataset_names = ['cae_lung_phantom_yolo_dataset', 'RobLUS_yolo_dataset']
    combined_dataset_name = 'combined_yolo_dataset'
    combined_dataset_dir = os.path.join(BASE_DIR, combined_dataset_name)
    
    print(f"Creating combined dataset from {dataset_names}...")
    
    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(combined_dataset_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(combined_dataset_dir, 'labels', split), exist_ok=True)
    
    # Copy images and labels from both datasets
    for dataset_name in dataset_names:
        dataset_root_dir = os.path.join(BASE_DIR, dataset_name)
        print(f"  Processing {dataset_name}...")
        
        for split in ['train', 'val']:
            src_img_dir = os.path.join(dataset_root_dir, 'images', split)
            dst_img_dir = os.path.join(combined_dataset_dir, 'images', split)
            src_label_dir = os.path.join(dataset_root_dir, 'labels', split)
            dst_label_dir = os.path.join(combined_dataset_dir, 'labels', split)
            
            if os.path.exists(src_img_dir):
                for img_file in os.listdir(src_img_dir):
                    src_img_path = os.path.join(src_img_dir, img_file)
                    dst_img_path = os.path.join(dst_img_dir, img_file)
                    if os.path.isfile(src_img_path):
                        shutil.copy2(src_img_path, dst_img_path)
            
            if os.path.exists(src_label_dir):
                for label_file in os.listdir(src_label_dir):
                    src_label_path = os.path.join(src_label_dir, label_file)
                    dst_label_path = os.path.join(dst_label_dir, label_file)
                    if os.path.isfile(src_label_path):
                        shutil.copy2(src_label_path, dst_label_path)
    
    # Create dataset.yaml for combined dataset
    yaml_data = {
        'path': os.path.abspath(combined_dataset_dir),
        'train': os.path.join('images', 'train'),
        'val': os.path.join('images', 'val'),
        'names': {0: 'pleural_line', 1: 'rib_shadow'}
    }
    
    data_yaml_path = os.path.join(combined_dataset_dir, 'dataset.yaml')
    with open(data_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
    
    print(f"Combined dataset ready at: {combined_dataset_dir}")
else:
    # Use single dataset
    dataset_name = 'RobLUS_yolo_dataset'
    dataset_root_dir = os.path.join(BASE_DIR, dataset_name)
    data_yaml_path = os.path.join(dataset_root_dir, 'dataset.yaml')

print(f'data path: {data_yaml_path}')

results = model.train(
    data=data_yaml_path,
    device='mps',
    epochs=50, 
    imgsz=640,
    batch=4,    
    patience=100,     
    name='combined_yolo11n' if USE_TWO_DATASETS else 'RobLUS_yolo11n',
)