# Transfer Learning for Computer Vision Assignment

## Overview
This assignment explores transfer learning for medical image classification on the PathMNIST dataset. We compare different architectures, fine-tuning strategies, and evaluate the data efficiency of transfer learning vs training from scratch.

## Key Findings
### Architecture Comparison
| Model          | Test Accuracy (%) | Parameters (M) |
|----------------|-------------------|----------------|
| ViT-B/16       | 92.83             | 85.81          |
| EfficientNet-B0| 90.71             | 4.02           |
| ResNet-50      | 87.51             | 23.53          |

### Fine-tuning Strategies
| Strategy                  | Test Accuracy (%) |
|---------------------------|-------------------|
| Full Fine-tuning (no aug) | 94.40             |
| Full Fine-tuning          | 93.11             |
| Feature Extraction        | 89.07             |

### Data Efficiency
| Training Data % | Pretrained Accuracy | From Scratch Accuracy |
|-----------------|---------------------|-----------------------|
| 100%            | 93.16%              | 75.67%                |
| 10%             | 92.06%              | 58.69%                |
| 5%              | 91.87%              | 49.48%                |

**Key Insight**: Even with only 5% of the training data, the pretrained model (91.87%) outperforms training from scratch on 100% of the data (75.67%)!

## Project Structure
```
CA1/
├── dataset.py              # Data loading and preprocessing
├── models.py               # Model definitions (ResNet, EfficientNet, ViT)
├── utils.py                # Training utilities and metrics
├── experiment1_architecture_comparison.py  # Architecture comparison
├── experiment2_finetune_strategies.py      # Fine-tuning strategies
├── experiment3_data_efficiency.py          # Data efficiency experiment
├── report.md               # Full assignment report
├── results/                # Training curves and results
└── requirements.txt        # Dependencies
```

## Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run experiments
python experiment1_architecture_comparison.py
python experiment2_finetune_strategies.py
python experiment3_data_efficiency.py
```

## Requirements
- PyTorch 2.0+
- torchvision
- medmnist
- matplotlib
- numpy
- tqdm

## Dataset
We use **PathMNIST** from MedMNIST, a medical imaging dataset with 107,180 H&E stained pathology images categorized into 9 tissue types.
