"""
Attention mechanisms for OCT-based Alzheimer's classification
MIRASOL @ MICCAI 2025
"""

from tensorflow.keras import layers
from config import ATTENTION_RATIO


def squeeze_excitation_block(input_tensor, ratio=ATTENTION_RATIO):
    """
    Squeeze-and-Excitation attention mechanism.
    
    Args:
        input_tensor: Input feature map
        ratio (int): Reduction ratio for squeeze operation
        
    Returns:
        tensor: Attention-weighted feature map
    """
    channels = input_tensor.shape[-1]
    
    # Squeeze: Global Average Pooling
    se = layers.GlobalAveragePooling2D()(input_tensor)
    
    # Excitation: Two fully connected layers
    se = layers.Reshape((1, 1, channels))(se)
    se = layers.Dense(channels // ratio, activation='relu', use_bias=False)(se)
    se = layers.BatchNormalization()(se)
    se = layers.Dense(channels, activation='sigmoid', use_bias=False)(se)
    
    # Scale: Element-wise multiplication
    return layers.multiply([input_tensor, se])


def attention_block(input_tensor, attention_type='se', **kwargs):
    """
    Generic attention block wrapper.
    
    Args:
        input_tensor: Input feature map
        attention_type (str): Type of attention mechanism ('se')
        **kwargs: Additional arguments for attention mechanism
        
    Returns:
        tensor: Attention-enhanced feature map
    """
    if attention_type == 'se':
        return squeeze_excitation_block(input_tensor, **kwargs)
    else:
        raise ValueError(f"Unsupported attention type: {attention_type}")