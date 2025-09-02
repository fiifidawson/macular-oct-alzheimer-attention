"""
Main training script for Macular OCT Alzheimer's Classification
MIRASOL @ MICCAI 2025

Paper: Attention-Enhanced Deep Learning for Multi-Class Alzheimer's Disease 
       Classification Using Macular OCT Images in Low-Resource Settings
"""

import os
import argparse
from src.data.data_loader import load_data
from src.models.models import build_attention_model, build_baseline_model
from src.training.train import train_model
from config import *


def main():
    parser = argparse.ArgumentParser(
        description="Train OCT-based Alzheimer's classification model"
    )
    parser.add_argument(
        '--model_type', 
        choices=['attention', 'baseline'], 
        default='attention',
        help='Model type to train (default: attention)'
    )
    parser.add_argument(
        '--dataset_path', 
        default=DATASET_PATH,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=EPOCHS,
        help=f'Number of training epochs (default: {EPOCHS})'
    )
    parser.add_argument(
        '--save_dir',
        default='models/',
        help='Directory to save trained models'
    )
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print("Loading and preprocessing data...")
    dataframes, generators, class_weights = load_data(args.dataset_path)
    train_df, val_df, test_df = dataframes
    train_gen, val_gen, test_gen = generators
    
    # Build model based on type
    print(f"Building {args.model_type} model...")
    if args.model_type == 'attention':
        model = build_attention_model()
        model_filename = 'alzheimers_effiB3_attention.h5'
    else:
        model = build_baseline_model()
        model_filename = 'alzheimers_effiB3_baseline.h5'
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Define save path
    model_save_path = os.path.join(args.save_dir, model_filename)
    
    # Train model
    print(f"\nTraining {args.model_type} model...")
    history = train_model(
        model=model,
        train_generator=train_gen,
        val_generator=val_gen,
        class_weights=class_weights,
        model_save_path=model_save_path,
        epochs=args.epochs
    )
    
    print(f"\n{'='*50}")
    print(f"Training completed successfully!")
    print(f"Model type: {args.model_type}")
    print(f"Model saved at: {model_save_path}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()