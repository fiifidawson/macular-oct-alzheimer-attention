"""
Configuration file for Macular OCT Alzheimer's Classification
MIRASOL @ MICCAI 2025
"""

import os

# Reproducibility
SEED = 42

# Image and Training Parameters
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 3

# Model Parameters
ATTENTION_RATIO = 16
LEARNING_RATE = 1e-5
DROPOUT_RATES = [0.4, 0.3]
DENSE_UNITS = [256, 128]

# Data Paths (modify according to your setup)
DATASET_PATH = "/path/to/Dataset"
MODEL_SAVE_PATH = "models/"

# Data Augmentation Parameters
AUGMENTATION_PARAMS = {
    'rotation_range': 20,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Class Mapping
ALZHEIMERS_MAP = {
    "NORMAL": "CN",
    "DRUSEN": "MCI",
    "CNV": "AD",
    "DME": "AD"
}

LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}

# Callback Parameters
REDUCE_LR_PARAMS = {
    'monitor': 'val_loss',
    'factor': 0.5,
    'patience': 3,
    'min_lr': 1e-7,
    'verbose': 1
}

EARLY_STOPPING_PARAMS = {
    'monitor': 'val_loss',
    'patience': 12,
    'restore_best_weights': True,
    'verbose': 1
}

MODEL_CHECKPOINT_PARAMS = {
    'monitor': 'val_auc',
    'save_best_only': True,
    'verbose': 1
}