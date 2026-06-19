# Animal Pose Estimation with PINN

A Physics-Informed Neural Network (PINN) for multi-species animal pose estimation, trained on the [Animal Kingdom dataset](https://github.com/sutdcv/Animal-Kingdom) (850+ species, 23 keypoints, 7 taxonomic classes).

## Overview

This project predicts 23 skeletal keypoints on animals across video frames using a ResNet-18-based encoder-decoder model (`PoseCNN`). Unlike standard pose estimators, the training loss incorporates physics-motivated constraints — bone length rigidity, temporal smoothness, and joint angle plausibility — computed over triplets of consecutive video frames.

At inference time, a YOLOv8 detector crops the animal from the frame before pose estimation, improving prediction quality on images where the animal occupies a small portion of the frame.

**Best validation PCK: 85.4%** (epoch 49/50)

## Pipeline

```
Video Frame → YOLOv8 Detection → Crop & Resize (256×256) → PoseCNN → 23 Keypoints
```

## Repository Structure

```
.
├── train.py            # Training loop with PINN loss, warm-up, checkpointing
├── dataset.py           # Triplet dataset construction from Animal Kingdom annotations
├── model.py              # PoseCNN architecture (ResNet-18 encoder + decoder)
├── losses.py             # PINN loss components (bone, smooth, angle) + PCK metric
├── best_model.pt         # Trained model weights (via Git LFS)
├── training_log.csv      # Per-epoch loss and metric log
└── README.md
```

## Model Architecture

- **Encoder:** ResNet-18 (ImageNet pretrained)
- **Decoder:** 4 transposed convolution layers
- **Output:** Heatmaps → soft-argmax → 23 keypoint coordinates
- **Parameters:** 13.9M total

## Loss Function

| Term | Description | Weight |
|------|-------------|--------|
| `L_data` | Visibility-masked MSE on keypoint coordinates | 1.0 |
| `L_bone` | Bone length rigidity across frames | 0.5 |
| `L_smooth` | Temporal smoothness across triplet | 0.1 |
| `L_angle` | Joint angle plausibility | 0.05 |

## Training Setup

- **Hardware:** Kaggle Tesla T4 GPU
- **Optimizer:** Adam, LR 1e-3 → cosine annealed, 5-epoch warm-up (head-only)
- **Batch size:** 32
- **Epochs:** 50
- **Triplets per epoch:** 5,000 (subsampled from 69,121 total)

## Usage

```bash
# Install dependencies
pip install torch torchvision pillow numpy ultralytics

# Train from scratch
python train.py --epochs 50 --batch_size 32 --num_workers 4

# Resume from checkpoint
python train.py --epochs 50 --batch_size 32 --num_workers 4 --resume
```

## Results

| Metric | Value |
|--------|-------|
| Best PCK | 85.4% |
| Best Val Loss | 0.0041 |
| Total Triplets | 69,121 |
| Species Covered | 850+ |

## Dataset & Resources

- [Training Notebook (Kaggle)](https://www.kaggle.com/code/bhavika2008/notebook8917a292bb)
- [Image Dataset (Kaggle)](https://www.kaggle.com/datasets/bhavika2008/ak-images)
- [Code & Annotations Dataset (Kaggle)](https://www.kaggle.com/datasets/bhavika2008/ak-code-ann)
- [Animal Kingdom Dataset (Original)](https://github.com/sutdcv/Animal-Kingdom)

## References

- Ng, K.Q., et al. (2022). *Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding*. CVPR 2022.
- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). *Physics-informed neural networks*. Journal of Computational Physics, 378, 686-707.
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
- Jocher, G., et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics

## Author

**Bhavika** — B.Tech, Computer Science Engineering, IIT (BHU) Varanasi

---

*This project was developed as an Exploratory Project at IIT (BHU) Varanasi.*
