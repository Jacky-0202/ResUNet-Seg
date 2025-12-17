# config.py

import torch
import os

# --- 1. Hardware Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if torch.cuda.is_available() else False
NUM_WORKERS = 8

# --- 2. Dataset Paths (LaPa) ---
# Root directory of the dataset
DATASET_ROOT = "/home/tec/Desktop/Project/Datasets/LaPa"
EXPERIMENT_NAME = "ResNet50_LaPa_Run1"

# Define explicit paths for Train, Val, and Test
# This structure assumes:
# LaPa/
#   ├── train/
#   │   ├── images/
#   │   └── labels/
#   ├── val/
#   │   ├── images/
#   │   └── labels/
#   └── test/ (Optional)

TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
VAL_DIR = os.path.join(DATASET_ROOT, 'val')

# Sub-folder names (Standard for LaPa)
IMG_DIR = 'images'       
MASK_DIR = 'labels'      

# Validation: Check if critical paths exist
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"❌ Training data not found: {TRAIN_DIR}")

if not os.path.exists(VAL_DIR):
    print(f"⚠️ Warning: Validation data not found at {VAL_DIR}. Training might fail if validation is required.")

# --- 3. Output Paths ---
# Get Project Root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', EXPERIMENT_NAME)
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# --- 4. Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 16       # LaPa is 512x512. If OOM, reduce to 4.
NUM_EPOCHS = 50
IMG_SIZE = 512       # LaPa standard resolution

# --- 5. Dataset Specifics ---
# Class Counting Rule:
# For Multi-class segmentation, Background is counted as a class.
# Formula: Total Classes = 1 (Background) + N (Objects)
NUM_CLASSES = 11
IGNORE_INDEX = None  # LaPa masks are clean integers (0-10)