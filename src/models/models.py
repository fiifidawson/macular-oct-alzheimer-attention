"""
Model architectures for OCT-based Alzheimer's classification
MIRASOL @ MICCAI 2025
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import AUC

from config import *
from .attention import attention_block


class SparseAUC(AUC):
    """Custom AUC metric for sparse categorical labels."""
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=NUM_CLASSES)
        super().update_state(y_true, y_pred, sample_weight)


def build_efficientnet_base():
    """
    Create EfficientNetB3 base model with selective layer unfreezing.
    
    Returns:
        keras.Model: Pre-trained EfficientNetB3 base
    """
    base = keras.applications.EfficientNetB3(
        include_top=False, 
        weights='imagenet', 
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Initially freeze all layers
    base.trainable = False
    
    # Unfreeze deeper blocks (5, 6, 7) for fine-tuning
    for layer in base.layers:
        if any(str(block_num) in layer.name for block_num in [5, 6, 7]):
            layer.trainable = True
    
    return base


def build_classification_head(x, use_attention=False):
    """
    Build classification head with optional attention mechanism.
    
    Args:
        x: Input tensor from base model
        use_attention (bool): Whether to include attention mechanism
        
    Returns:
        tensor: Output logits
    """
    # Optional attention mechanism
    if use_attention:
        x = attention_block(x, attention_type='se')
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # First dense layer (512 units as per published paper)
    x = layers.Dense(DENSE_UNITS[0], use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(DROPOUT_RATES[0])(x)
    
    # Second dense layer  
    x = layers.Dense(DENSE_UNITS[1], use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(DROPOUT_RATES[1])(x)
    
    # Output layer
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return outputs


def build_model(use_attention=False, model_name="efficientnet_oct"):
    """
    Build complete model architecture.
    
    Args:
        use_attention (bool): Whether to include attention mechanism
        model_name (str): Model name for identification
        
    Returns:
        keras.Model: Compiled model
    """
    # Build base model
    base = build_efficientnet_base()
    
    # Build classification head
    outputs = build_classification_head(base.output, use_attention=use_attention)
    
    # Create complete model
    model = keras.Model(inputs=base.input, outputs=outputs, name=model_name)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            SparseAUC(name='auc'),
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')
        ]
    )
    
    return model


def build_attention_model():
    """Build model with attention mechanism."""
    return build_model(use_attention=True, model_name="efficientnet_oct_attention")


def build_baseline_model():
    """Build baseline model without attention."""
    return build_model(use_attention=False, model_name="efficientnet_oct_baseline")