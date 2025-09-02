"""
Training pipeline for OCT-based Alzheimer's classification
MIRASOL @ MICCAI 2025
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import *


def setup_environment():
    """Setup reproducible training environment."""
    tf.random.set_seed(SEED)
    np.random.seed(SEED)


def create_callbacks(model_save_path):
    """
    Create training callbacks.
    
    Args:
        model_save_path (str): Path to save the best model
        
    Returns:
        list: List of keras callbacks
    """
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(**REDUCE_LR_PARAMS),
        keras.callbacks.EarlyStopping(**EARLY_STOPPING_PARAMS),
        keras.callbacks.ModelCheckpoint(model_save_path, **MODEL_CHECKPOINT_PARAMS)
    ]
    return callbacks


def train_model(model, train_generator, val_generator, class_weights, 
                model_save_path, epochs=EPOCHS):
    """
    Train the model.
    
    Args:
        model: Compiled keras model
        train_generator: Training data generator
        val_generator: Validation data generator
        class_weights (dict): Class weights for handling imbalance
        model_save_path (str): Path to save the model
        epochs (int): Number of training epochs
        
    Returns:
        keras.callbacks.History: Training history
    """
    # Setup environment
    setup_environment()
    
    # Create callbacks
    callbacks = create_callbacks(model_save_path)
    
    # Train model
    print(f"Starting training for {epochs} epochs...")
    print(f"Model will be saved to: {model_save_path}")
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(model_save_path.replace('.h5', '_final.h5'))
    print(f"Training completed! Model saved successfully.")
    
    return history


def load_trained_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to saved model
        
    Returns:
        keras.Model: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    return model