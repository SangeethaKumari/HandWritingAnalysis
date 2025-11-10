# Skin Disease Model Download Guide

## Quick Start

The easiest way to get started is to create a placeholder model:

```bash
cd /home/sangeethagsk/agent_bootcamp/HandWritingAnalysis
python src/imageanalysis/train_skin_model.py
```

Choose option 1 or 2 to create a basic model file.

## Where to Download Pre-trained Models

### Option 1: Kaggle Datasets
1. Visit [Kaggle.com](https://www.kaggle.com)
2. Search for "skin disease classification" or "dermatology dataset"
3. Popular datasets:
   - HAM10000 (Human Against Machine)
   - Skin Cancer Classification datasets
   - Dermatology Image datasets

### Option 2: ISIC Archive
- **Website**: https://www.isic-archive.com
- **Description**: International Skin Imaging Collaboration
- Contains thousands of dermatoscopic images
- Free to download with registration

### Option 3: GitHub Repositories
Search GitHub for:
- "skin disease classification pytorch"
- "dermatology deep learning"
- "skin lesion classification"

Example repositories:
- https://github.com/search?q=skin+disease+classification+pytorch

### Option 4: Research Papers
Many research papers provide model weights:
- Papers with Code: https://paperswithcode.com
- Search for "skin disease classification" or "dermatology CNN"

## Training Your Own Model

If you have a dataset organized like this:
```
dataset/
  ├── Acne/
  │   ├── image1.jpg
  │   └── image2.jpg
  ├── Eczema/
  ├── Psoriasis/
  ├── Urticaria/
  └── Rosacea/
```

You can train a model using PyTorch's transfer learning with ImageNet pre-trained weights.

## Quick Fix: Use ImageNet Pre-trained Model

Run the training script and choose option 2:
```bash
python src/imageanalysis/train_skin_model.py
```

This will create a model using ImageNet weights (good starting point, but needs fine-tuning for accurate skin disease classification).

## Important Notes

⚠️ **Warning**: A placeholder or untrained model will give random/inaccurate predictions. For real medical use, you need a properly trained model with validated accuracy.

For testing purposes, the placeholder model will work, but predictions will not be meaningful.

