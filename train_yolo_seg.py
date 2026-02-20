import os
from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt')

dataset_name = 'cae_lung_phantom_yolo_dataset'

dataset_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_name)
data_yaml_path = os.path.join(dataset_root_dir, 'dataset.yaml')

print(f'data path: {data_yaml_path}')

results = model.train(
    data=data_yaml_path,
    epochs=50, 
    imgsz=640,
    batch=4,    
    patience=100,     
    name='cae_lung_phantom_yolo11n',
)