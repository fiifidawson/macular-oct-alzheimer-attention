"""
Data loading and preprocessing utilities
MIRASOL @ MICCAI 2025
"""

import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import compute_class_weight
from config import *


def build_dataframe(split_name, dataset_path=DATASET_PATH):
    """
    Reads image files under train/val/test splits into a DataFrame.
    
    Args:
        split_name (str): 'train', 'val', or 'test'
        dataset_path (str): Path to dataset directory
        
    Returns:
        pd.DataFrame: DataFrame with 'filepath' and 'label' columns
    """
    records = []
    split_dir = os.path.join(dataset_path, split_name)
    
    for orig_cat, new_cat in ALZHEIMERS_MAP.items():
        dir_path = os.path.join(split_dir, orig_cat)
        if not os.path.isdir(dir_path):
            continue
            
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                records.append({
                    'filepath': os.path.join(dir_path, fname),
                    'label': LABEL_MAP[new_cat]
                })
    
    return pd.DataFrame(records)


def create_data_generators(train_df, val_df, test_df=None):
    """
    Create ImageDataGenerators for training, validation, and optionally test sets.
    
    Args:
        train_df (pd.DataFrame): Training data DataFrame
        val_df (pd.DataFrame): Validation data DataFrame  
        test_df (pd.DataFrame, optional): Test data DataFrame
        
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    # Training generator with augmentation
    train_gen = ImageDataGenerator(
        rescale=1./255,
        **AUGMENTATION_PARAMS
    ).flow_from_dataframe(
        train_df, 
        x_col='filepath', 
        y_col='label',
        target_size=IMG_SIZE, 
        class_mode='raw',
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        seed=SEED
    )
    
    # Validation generator (no augmentation)
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        val_df, 
        x_col='filepath', 
        y_col='label',
        target_size=IMG_SIZE, 
        class_mode='raw',
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    # Test generator (if provided)
    test_gen = None
    if test_df is not None:
        test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
            test_df, 
            x_col='filepath', 
            y_col='label',
            target_size=IMG_SIZE, 
            class_mode='raw', 
            shuffle=False
        )
    
    return train_gen, val_gen, test_gen


def compute_class_weights(train_df):
    """
    Compute class weights for handling class imbalance.
    
    Args:
        train_df (pd.DataFrame): Training data DataFrame
        
    Returns:
        dict: Class weights dictionary
    """
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    return dict(enumerate(weights))


def load_data(dataset_path=DATASET_PATH):
    """
    Load and prepare all data splits.
    
    Args:
        dataset_path (str): Path to dataset directory
        
    Returns:
        tuple: DataFrames and generators for train/val/test splits
    """
    # Build DataFrames
    train_df = build_dataframe('train', dataset_path)
    val_df = build_dataframe('val', dataset_path)
    test_df = build_dataframe('test', dataset_path)
    
    # Create generators
    train_gen, val_gen, test_gen = create_data_generators(train_df, val_df, test_df)
    
    # Compute class weights
    class_weights = compute_class_weights(train_df)
    
    print("Training samples per class:")
    print(train_df['label'].value_counts())
    print("Class weights:", class_weights)
    
    return (train_df, val_df, test_df), (train_gen, val_gen, test_gen), class_weights