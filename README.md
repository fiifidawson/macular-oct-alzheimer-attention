# Attention-Enhanced Deep Learning for Multi-Class Alzheimer's Disease Classification Using Macular OCT Images

[![MICCAI](https://img.shields.io/badge/MICCAI-MIRASOL%202025-blue)](https://conferences.miccai.org/2025/en/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange)](https://tensorflow.org)

## Overview

This repository contains the implementation for our paper presented at the Medical Image Computing in Resource Constrained Settings Workshop & Knowledge Interchange (MIRASOL) at MICCAI 2025.

**Abstract**: We present an attention-enhanced deep learning approach for multi-class Alzheimer's disease classification using macular optical coherence tomography (OCT) images, specifically designed for low-resource clinical settings.

## Key Features

- **EfficientNetB3-based architecture** with selective fine-tuning
- **Squeeze-and-Excitation attention mechanism** for enhanced feature representation  
- **Multi-class classification**: Cognitive Normal (CN), Mild Cognitive Impairment (MCI), Alzheimer's Disease (AD)
- **Class imbalance handling** with computed class weights
- **Data augmentation** for improved generalization

## Dataset

The model is trained on retinal OCT images with the following mapping:
- `NORMAL` → `CN` (Cognitively Normal)
- `DRUSEN` → `MCI` (Mild Cognitive Impairment)  
- `CNV` → `AD` (Alzheimer's Disease)
- `DME` → `AD` (Alzheimer's Disease)

<!-- ## Repository Structure

```
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration parameters
├── main.py                     # Main training script
├── src/
│   ├── data/
│   │   └── data_loader.py     # Data loading and preprocessing
│   ├── models/
│   │   ├── attention.py       # Attention mechanisms
│   │   └── models.py          # Model architectures
│   └── training/
│       └── train.py           # Training pipeline
└── models/                     # Saved model directory
``` -->

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fiifidawson/macular-oct-alzheimer-attention.git
cd macular-oct-alzheimer-attention
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the attention-enhanced model:
```bash
python main.py --model_type attention --dataset_path /path/to/dataset
```

Train the baseline model (without attention):
```bash
python main.py --model_type baseline --dataset_path /path/to/dataset
```

### Additional Options

```bash
python main.py --help
```

Options:
- `--model_type`: Choose between 'attention' or 'baseline'
- `--dataset_path`: Path to your OCT dataset
- `--epochs`: Number of training epochs (default: 20)
- `--save_dir`: Directory to save trained models

## Model Architecture

### Base Architecture
- **Backbone**: EfficientNetB3 pre-trained on ImageNet
- **Fine-tuning**: Selective unfreezing of blocks 5, 6, and 7
- **Input size**: 300×300×3 RGB images

### Attention Mechanism
- **Type**: Squeeze-and-Excitation (SE) attention
- **Reduction ratio**: 16
- **Integration**: Applied after backbone feature extraction

### Classification Head
- Global Average Pooling
- Dense layers: 256 → 128 → 3 (with BatchNorm, ReLU, Dropout)
- Output: Softmax activation for 3-class classification

<!-- ## Configuration

Key parameters can be modified in `config.py`:

```python
# Model parameters
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-5

# Attention parameters
ATTENTION_RATIO = 16

# Architecture parameters
DENSE_UNITS = [256, 128]
DROPOUT_RATES = [0.4, 0.3]
``` -->

## Data Format

Expected directory structure:
```
dataset/
├── train/
│   ├── NORMAL/
│   ├── DRUSEN/
│   ├── CNV/
│   └── DME/
├── val/
│   ├── NORMAL/
│   ├── DRUSEN/
│   ├── CNV/
│   └── DME/
└── test/
    ├── NORMAL/
    ├── DRUSEN/
    ├── CNV/
    └── DME/
```

<!-- ## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{yourname2025attention,
  title={Attention-Enhanced Deep Learning for Multi-Class Alzheimer's Disease Classification Using Macular OCT Images in Low-Resource Settings},
  author={Your Name and Co-authors},
  booktitle={Medical Image Computing in Resource Constrained Settings Workshop (MIRASOL) at MICCAI},
  year={2025}
}
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{dawson2025attention,
  title={Attention-Enhanced Deep Learning for Multi-Class Alzheimer's Disease Classification Using Macular OCT Images in Low-Resource Settings},
  author={Dawson, Edem Fiifi and Darko, Louis and Tagoe, Hephzi and Ahiabor, Edem and Oteng, Kwame and Gyamfi, Samuel},
  booktitle={Medical Image Computing in Resource Constrained Settings Workshop (MIRASOL) at MICCAI},
  year={2025}
}
``` -->

## Ethical Considerations

This research was conducted with proper ethical approvals:
- **Korle-Bu Teaching Hospital** institutional review board approval
- **Emmanuel Eye Medical Centre** ethics committee approval  
- **Data de-identification** following local research ethics guidelines
<!-- - **International standards** compliance for medical data handling -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MICCAI - MIRASOL Workshop organizers
- Korle-Bu Teaching Hospital, Lions International Eye Centre
- Emmanuel Eye Medical Centre  
- Academic City University, Department of Biomedical Engineering

## Contact

For questions or collaborations, please reach out to:
- **Edem Fiifi Dawson**: edem.dawson@acity.edu.gh
- **Louis Darko**: louis.darko@acity.edu.gh  
- **Hephzi Tagoe**: hephzi.tagoe@acity.edu.gh

---

**Important Clinical Note**: While this work demonstrates promising results for OCT-based AD screening, OCT alone is not yet a validated standalone diagnostic test for Alzheimer's Disease. This research highlights the potential for enhancing diagnostic capacity in under-resourced healthcare settings and should be integrated with comprehensive clinical assessment.