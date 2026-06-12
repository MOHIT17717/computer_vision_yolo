import os
import urllib.request
import cv2
import yaml
from ultralytics import YOLO

# Create dataset structure
dataset_dir = os.path.join(os.getcwd(), 'datasets', 'garbage_bins')
images_dir = os.path.join(dataset_dir, 'images', 'train')
labels_dir = os.path.join(dataset_dir, 'labels', 'train')

os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Generate data.yaml
data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
data_yaml = {
    'path': dataset_dir,
    'train': 'images/train',
    'val': 'images/train',
    'names': {
        0: 'normal_bin',
        1: 'overflowing_bin'
    }
}
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_yaml, f)

# Download sample images and create synthetic bounding boxes
# (x_center, y_center, width, height) = (0.5, 0.5, 0.8, 0.8) for all as a fallback
image_urls = [
    # Normal Bins (Class 0)
    ("https://images.unsplash.com/photo-1595278474246-17b189ff70a5?w=640", 0),
    ("https://images.unsplash.com/photo-1611284446314-60a58ac0deb9?w=640", 0),
    ("https://images.unsplash.com/photo-1532996122724-e3c354a0b15b?w=640", 0),
    
    # Overflowing Bins (Class 1) - using generic messy/trash images
    ("https://images.unsplash.com/photo-1605600659873-d808a13e4d2a?w=640", 1),
    ("https://images.unsplash.com/photo-1530587191325-3db32d826c18?w=640", 1),
    ("https://images.unsplash.com/photo-1550989460-0adf9ea622e2?w=640", 1)
]

print("Downloading dataset and generating YOLO labels...")
for i, (url, cls_id) in enumerate(image_urls):
    img_path = os.path.join(images_dir, f"img_{i}.jpg")
    label_path = os.path.join(labels_dir, f"img_{i}.txt")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open(img_path, 'wb') as f:
                f.write(response.read())
                
        # Generate bounding box taking up central 80% of image
        with open(label_path, 'w') as f:
            f.write(f"{cls_id} 0.5 0.5 0.8 0.8\n")
            
        print(f"Prepared image {i} (Class {cls_id})")
    except Exception as e:
        print(f"Failed to download image {i}: {e}")

# Train the model
print("\n🚀 Starting Custom YOLOv8 Training...")
model = YOLO('yolov8n.pt')

# Train for 5 epochs on CPU (this is very fast for 6 images)
results = model.train(
    data=data_yaml_path,
    epochs=5,
    imgsz=320,
    device='cpu',
    project='runs/detect',
    name='train_overflow',
    exist_ok=True
)

print("\n✅ Training Complete!")
print(f"Custom model saved to: {os.path.join(os.getcwd(), 'runs', 'detect', 'train_overflow', 'weights', 'best.pt')}")
