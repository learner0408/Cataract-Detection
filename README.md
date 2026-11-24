# Cataract-Detection

Deep learning solution that classifies retinal fundus images as **Cataract** or **Normal** using TensorFlow/Keras. The repository contains two Jupyter notebooks:

| Notebook | Description |
|---|---|
| `CD_Modelling.ipynb` | Compares four models — a logistic-regression baseline, VGG-16 built from scratch, and transfer learning with MobileNetV2 and ResNet50 |
| `Cataract-Detection-Novel-CNN-Architecture .ipynb` | Trains a novel custom CNN architecture on the ODIR-5K dataset |

## Dataset

- **CD_Modelling.ipynb** — a custom fundus image dataset (loaded from Google Drive as `CD_Dataset_Krishna`):
  - **8,068 training images** and **1,600 test images**
  - Two classes: *Cataract* and *Normal*
- **Novel CNN notebook** — the [ODIR-5K (Ocular Disease Recognition)](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) dataset from Kaggle:
  - Cataract images selected using the diagnostic keywords; balanced with 600 sampled normal-fundus images
  - All images resized to 224×224 and split 80/20 into train/test

## Models Used

1. **Logistic Regression baseline** — `Flatten` + `Dense(2, sigmoid)`, trained with SGD
2. **VGG-16** — built from scratch (13 conv layers + max pooling)
3. **MobileNetV2** *(transfer learning)* — ImageNet-pretrained, frozen base + `GlobalAveragePooling2D` + `Dense(2)` head
4. **ResNet50** *(transfer learning)* — ImageNet-pretrained, frozen base + `GlobalMaxPooling2D` + `Dense(2)` head
5. **Novel CNN architecture** — five convolutional blocks (1×1 and 3×3 convolutions, 32→128 filters) with BatchNormalization, ReLU, and MaxPooling, followed by `Dense(1024) → Dense(128) → softmax(2)`

## Training Characteristics

- **Input size:** 100×100 (baseline), 160×160 (MobileNetV2), 220×220 (ResNet50), 224×224 (VGG-16 and novel CNN)
- **Optimizer:** SGD for the baseline; Adam for the rest (lr 0.01 for VGG-16, 0.001 for pretrained models, 1e-6 for the novel CNN)
- **Loss:** binary / sparse categorical crossentropy
- **Batch size:** 16–32; **Epochs:** 10 (pretrained), 25 (VGG-16), 30 (baseline), 1000 (novel CNN)
- **Augmentation** (MobileNetV2 path): horizontal/vertical flips, 30° rotations, width shifts
- **Regularization:** L1 activity regularizer (novel CNN), Dropout 0.2 / 0.5, BatchNormalization

## Results

| Model | Validation Accuracy |
|---|---|
| Logistic Regression baseline | ~50% (no better than chance) |
| VGG-16 (from scratch, lr 0.01) | ~50% (failed to converge) |
| ResNet50 (transfer learning) | ~95% peak, unstable (~75% final) |
| MobileNetV2 (transfer learning) | **97.25%** |
| **Novel CNN (ODIR-5K)** | **99.16%** |

**Takeaway:** transfer learning with MobileNetV2 gives the best stable result on the custom dataset, while the novel CNN architecture reaches **99.16% accuracy** on the ODIR-5K dataset — the strongest overall performance in this project.
