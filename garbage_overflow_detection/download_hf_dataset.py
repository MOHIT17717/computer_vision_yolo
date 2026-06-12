import os
import yaml
from datasets import load_dataset
from PIL import Image

def download_and_format_dataset():
    print("Downloading dataset from HuggingFace...")
    # Load the dataset
    ds = load_dataset("keremberke/garbage-object-detection", "full")
    
    base_dir = os.path.join(os.path.dirname(__file__), "data", "hf_garbage")
    os.makedirs(base_dir, exist_ok=True)
    
    # We will just map the categories provided in the dataset
    # keremberke/garbage-object-detection has categories: 
    # ['biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
    # Wait, the user specifically asked for 'dustbin'. Does this dataset have bins?
    # Let's inspect the categories first.
    features = ds['train'].features['objects']
    print(features)

if __name__ == "__main__":
    download_and_format_dataset()
