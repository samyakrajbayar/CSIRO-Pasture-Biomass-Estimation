# CSIRO - Image2Biomass Prediction

## Overview

This repository contains a solution for the **CSIRO - Image2Biomass Prediction** Kaggle competition. The task is to predict five key pasture biomass components from top‑view RGB images:

| Target | Description |
|--------|-------------|
| `Dry_Green_g` | Dry green vegetation (excluding clover) |
| `Dry_Dead_g` | Dry dead material |
| `Dry_Clover_g` | Dry clover biomass |
| `GDM_g` | Green Dry Matter (Green + Clover) |
| `Dry_Total_g` | Total dry biomass (GDM + Dead) |

The evaluation metric is a **globally weighted R²**, with per‑target weights: `Dry_Green_g`: 0.1, `Dry_Dead_g`: 0.1, `Dry_Clover_g`: 0.1, `GDM_g`: 0.2, `Dry_Total_g`: 0.5.

---

## Data

- **Training data**: 361 images (JPEG) with associated metadata (Sampling Date, State, Species, NDVI, height) and ground‑truth biomass values.
- **Test data**: Over 800 images; predictions are required for each `(image, target_name)` pair.
- **Data split**: A stratified 5‑fold split by `State` and `Sampling_Date` is used (provided by [CSIRO‑DataSplit](https://www.kaggle.com/code/samu2505/csiro-datasplit)).
- **Citation**: If you use this dataset, please cite the accompanying paper:
  ```
  @misc{liao2025estimatingpasturebiomasstopview,
    title={Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture},
    author={Qiyu Liao and Dadong Wang and Rebecca Haling and Jiajun Liu and Xun Li and Martyna Plomecka and Andrew Robson and Matthew Pringle and Rhys Pirie and Megan Walker and Joshua Whelan},
    year={2025},
    eprint={2510.22916},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
  }
  ```

---

## Models Used

### 1. SigLIP (Vision Transformer)

- **Model**: `google/siglip-so400m-patch14-384`
- **Role**: Extracts high‑level visual embeddings from image patches.
- **Implementation**: The image is split into overlapping patches (`520×520` with `16` px overlap), embeddings are averaged per image, and PCA/PLS/GMM are applied to create a compact feature set.
- **Source**: [Kaggle Model](https://www.kaggle.com/models/aishikai/google-siglip-so400m-patch14-384/Transformers/default/1)

### 2. DINOv3 (Vision Transformer)

- **Model**: `vit_large_patch16_dinov3_qkvb` (ViT‑Large)
- **Role**: Processes the left and right halves of each image separately, then fuses features via a FiLM (Feature‑wise Linear Modulation) layer.
- **Training**: 5‑fold cross‑validation with EMA, gradient checkpointing, and cosine scheduling.
- **Weights**: Pretrained backbone from [timm](https://github.com/huggingface/pytorch-image-models) (version `1.0.22`).
- **Source**: [Kaggle Model](https://www.kaggle.com/models/giovannyrodrguez/modelv3/PyTorch/default/1) (contains 5 fold‑trained `.pth` files)

### 3. Ensemble of Regressors (on SigLIP features)

- **Models**: HistGradientBoosting, GradientBoosting, CatBoost, LightGBM
- **Features**: PCA + PLS + GMM + semantic text‑derived features (from SigLIP’s text encoder)
- **Training**: 5‑fold cross‑validation with target‑wise scaling and post‑processing (mass balance enforcement).

---

## Methodology

### Feature Extraction

1. **SigLIP Embeddings**  
   - Each image is tiled into overlapping patches; embeddings are averaged to obtain a 1152‑dim vector.
2. **Semantic Features**  
   - Text prompts (e.g., “dense tall pasture”, “dry brown dead grass”) are encoded with SigLIP’s text encoder.
   - Cosine similarity between image embeddings and text prototypes yields 11 semantic scores (including ratios like `green/(green+dead)`).
3. **Supervised Embedding Engine**  
   - PCA (80% variance), PLS (8 components), and GMM (6 components) are applied to the embeddings.
   - These, together with the semantic features, form the final input for the gradient‑boosted models.

### DINOv3 Model

- **Architecture**: ViT‑Large backbone + FiLM fusion of left/right image halves + three regression heads (Green, Clover, Dead) with Softplus activation.
- **Training**:
  - 5‑fold stratified split by `State` and `Sampling_Date`.
  - Mixed precision (bfloat16), gradient accumulation, and EMA.
  - Learning rate: `5e‑4` for backbone, `1e‑3` for heads; cosine warmup.
  - Early stopping with patience `5`.
- **Inference**: Test‑time augmentation (TTA) with 7 views (original, horizontal/vertical flips, brightness/contrast adjustments, rotation).

### Post‑Processing & Ensemble

- **Mass Balance Enforcement**: Orthogonal projection ensures `GDM = Green + Clover` and `Total = GDM + Dead`.
- **Smart Thresholding**: Values below target‑specific thresholds (e.g., `0.15` for Clover) are set to zero.
- **Temperature Scaling**: Reduces overconfidence by dividing predictions by `1.12–1.15`.
- **Ensemble**:
  - **SigLIP‑based** predictions (from gradient‑boosted models) and **DINOv3** predictions are blended with target‑specific weights (e.g., DINO gets higher weight for Clover and Dead).
  - The final submission is a weighted geometric mean followed by physics‑based constraints.

---

## Results

The ensemble achieves a **weighted R² of ~0.75–0.77** on the private leaderboard, with strong performance across all five targets.

---

## How to Run

### Dependencies

Install the required packages (use Python 3.10+):

```bash
pip install -U --no-deps timm==1.0.22
pip install torch torchvision pandas numpy scikit-learn lightgbm catboost opencv-python pillow transformers tqdm scipy albumentations
```

### Data Preparation

1. Download the competition data from [Kaggle](https://www.kaggle.com/competitions/csiro-biomass/data) and place it in `/kaggle/input/csiro-biomass/`.
2. The data split CSV (`csiro_data_split.csv`) is expected at `/kaggle/input/csiro-datasplit/csiro_data_split.csv`.

### Model Weights

- **SigLIP**: Automatically downloaded via `transformers` (or use the Kaggle model link).
- **DINOv3 fold weights**: Place the 5 `.pth` files from [modelv3](https://www.kaggle.com/models/giovannyrodrguez/modelv3/PyTorch/default/1) in `/kaggle/input/modelv3/pytorch/default/1/models_retrained/`.

### Running the Notebook

Execute the cells in `CSIRO.ipynb` sequentially:

1. **Data loading & preprocessing** – pivots the long‑format training data.
2. **SigLIP embedding extraction** – computes image and semantic features.
3. **Training of gradient‑boosted models** (5‑fold CV) – produces `submission_siglip.csv`.
4. **DINOv3 inference** – runs `csiro_infer.py` to produce `submission_dinov2026.csv`.
5. **Ensemble & post‑processing** – blends the two submissions, applies mass balance, thresholds, and scaling, and outputs `submission.csv`.

> **Note**: The DINOv3 training loop is included but commented out; the notebook loads pre‑trained fold weights for inference only.

---

## File Structure

```
.
├── CSIRO.ipynb               # Main notebook (all steps)
├── csiro_infer.py            # Standalone DINOv3 inference script
├── submission_siglip.csv     # Predictions from SigLIP + boosted models
├── submission_dinov2026.csv  # Predictions from DINOv3 (with TTA)
├── submission.csv            # Final ensemble submission
└── README.md                 # This file
```

---

## License

This project is for educational/research purposes. The competition data is subject to Kaggle’s terms. The code is released under the Apache 2.0 license (as per the data split notebook).

---

## Acknowledgements

- **CSIRO** for the dataset and the challenge.
- **Kaggle** for the competition platform.
- **Hugging Face** & **timm** for pretrained models.
- All contributors to the open‑source libraries used.

---

## Contact

For questions or suggestions, please open an issue or reach out via the Kaggle competition discussion board.
