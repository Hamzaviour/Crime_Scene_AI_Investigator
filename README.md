# AI-Based Crime Scene Analysis — Phase 1

Violence detection from CCTV and real-world video using deep learning. Phase 1 covers end-to-end preprocessing, training of three candidate models, and test-set evaluation with comparison metrics.

**Team**

| Member | Role |
|--------|------|
| Rameen Babar | Preprocessing & frame extraction |
| Rida Fatima | Model training |
| Hamza | Model Training & Test evaluation & model comparison |

---

## Problem

Binary classification: **Violence** vs **Non-Violence** from short video clips.

- Input: 16 evenly spaced frames per video, resized to 224×224, normalized to `[0, 1]`
- Output: Single sigmoid probability (threshold 0.5 for class label)

---

## Datasets

Two public datasets are merged into one unified metadata table:

| Dataset | Source | Structure |
|---------|--------|-----------|
| **RLVD** — Real Life Violence Situations | [Kaggle](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) | Flat: `ClassName/video.mp4` |
| **SCVD** — SmartCity CCTV Violence Detection | [Kaggle](https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd) | Split: `Split/ClassName/video.avi` |

**Total indexed videos:** 2,000 (balanced 1,000 / 1,000 after binary label mapping)

Datasets are **not** stored in this repo due to size. See [`data/README.md`](data/README.md) for download links.

---

## Project Structure

```
CV/
├── README.md                          # This file
├── data/
│   └── README.md                      # Dataset download links
└── notebooks/
    ├── phase1-preprocessing-pipeline.ipynb.ipynb   # Main notebook (full pipeline)
    ├── dataset_metadata.csv           # Unified video index (2,000 rows)
    ├── train_split.csv                # Training set (1,400 videos)
    ├── val_split.csv                  # Validation set (300 videos)
    ├── best_MobileNetV2_BiLSTM.h5      # Trained checkpoint — Model 1
    ├── best_C3D_Lite.h5               # Trained checkpoint — Model 2
    ├── best_ConvLSTM2D.keras          # Trained checkpoint — Model 3
    ├── training_histories.npy         # Validation curves (MobileNetV2, C3D Lite)
    ├── model_comparison_test_metrics.csv   # Generated after evaluation (Section 9)
    └── test_predictions.npy         # Generated after evaluation (Section 9)
```

---

## Models

Three architectures were trained on the same train/val/test split:

| Model | Description | Checkpoint |
|-------|-------------|------------|
| **MobileNetV2 + Bi-LSTM** | Pretrained MobileNetV2 per frame → Bi-LSTM temporal pooling | `best_MobileNetV2_BiLSTM.h5` |
| **C3D Lite** | Lightweight 3D CNN for spatiotemporal features | `best_C3D_Lite.h5` |
| **ConvLSTM2D** | End-to-end ConvLSTM (frames resized to 96×96 internally) | `best_ConvLSTM2D.keras` |

**Training config (shared)**

- `NUM_FRAMES = 16`, `FRAME_SIZE = (224, 224)`, `BATCH_SIZE = 4`
- `EPOCHS = 9` (GPU) / `3` (CPU fallback)
- Class-weighted loss for imbalance handling
- Callbacks: EarlyStopping (`val_auc`), ReduceLROnPlateau, ModelCheckpoint

**Best validation metrics (from training)**

| Model | Best Val AUC | Best Val Accuracy |
|-------|--------------|-------------------|
| MobileNetV2 + Bi-LSTM | 0.988 | 0.950 |
| C3D Lite | 0.956 | 0.886 |
| ConvLSTM2D | 0.972 | 0.913 |

---

## Data Split

Stratified split on binary label (`SEED = 42`):

| Split | Videos | Share |
|-------|--------|-------|
| Train | 1,400 | 70% |
| Validation | 300 | 15% |
| Test | 300 | 15% |

All three models are evaluated on the **same 300-video test set** so results are directly comparable.

---

## Notebook Pipeline

Open `notebooks/phase1-preprocessing-pipeline.ipynb.ipynb` on **Kaggle** (free GPU recommended).

| Section | Content |
|---------|---------|
| 1–2 | Imports, config, dataset paths |
| 3 | `preprocess_video()` — frame extraction |
| 4 | Unified metadata across RLVD + SCVD |
| 5 | Train / val / test split |
| 6 | `VideoDataGenerator` (on-the-fly decoding) |
| 7 | Class weights |
| 8 | ConvLSTM2D architecture + training |
| **9** | **Test evaluation & model comparison (Hamza)** |

### Running on Kaggle

1. Create a Kaggle account and **New Notebook**.
2. **File → Import Notebook** and upload the notebook file.
3. **+ Add Input** — add both datasets:
   - `mohamedmustafa/real-life-violence-situations-dataset`
   - `toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd`
4. **Settings → Accelerator → GPU T4 x2**
5. Upload the three model checkpoints into the notebook working directory (or keep them alongside the notebook if already present).
6. **Run All**, or run Sections 1–16 then **Section 9** only if models are already trained.

> Evaluation requires access to the raw video files (Kaggle input paths). It will not run locally unless you mirror the datasets and update `RLVD_PATH` / `SCVD_PATH`.

---

## Evaluation Metrics (Section 9)

For each model on the held-out test set:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **AUC-ROC**
- **Confusion matrix**
- **Classification report**

**Visual outputs**

- Per-model confusion matrices
- Overlaid ROC curves (all three models)
- Side-by-side metric bar charts + heatmap
- Validation training curves from `training_histories.npy`

**Saved artefacts**

- `model_comparison_test_metrics.csv` — comparison table sorted by AUC
- `test_predictions.npy` — per-model `y_true`, `y_pred`, `y_prob`, confusion matrices

---

## Requirements

```
tensorflow >= 2.x
opencv-python (cv2)
numpy
pandas
matplotlib
scikit-learn
```

Kaggle notebooks ship with these pre-installed. For local use, install via:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
```

---

## Next Steps (Phase 2+)

- Crime-scene-specific fine-tuning or transfer to deployment pipeline
- Real-time inference on CCTV streams
- Model selection based on test-set comparison (accuracy vs latency vs size)
- Export best model to TFLite / ONNX for edge deployment

---

## License & Attribution

- **RLVD:** [Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) — cite the original authors when publishing results.
- **SCVD:** [SmartCity CCTV Violence Detection Dataset](https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd) — cite the original authors when publishing results.
