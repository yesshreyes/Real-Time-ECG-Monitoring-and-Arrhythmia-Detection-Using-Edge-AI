# Three-Stage ECG Arrhythmia Detection System

Implementation plan for building a hierarchical ECG classification system optimized for edge deployment.

## Overview

This system implements a **three-stage cascade architecture**:
- **Stage 0 (SQI)**: Ultra-lightweight signal quality filter (Good vs Bad) - quality screening
- **Stage 1**: Lightweight binary classifier (Normal vs Abnormal) - high sensitivity screening
- **Stage 2**: Multi-class classifier (Supraventricular vs Ventricular arrhythmia) - precise diagnosis

The models will be trained on the MIT-BIH Arrhythmia Dataset and optimized for edge deployment using TensorFlow Lite quantization.

---

## Dataset Analysis

**MIT-BIH Arrhythmia Dataset Structure:**
- 48 patient records (IDs: 100-234)
- Sampling frequency: 360 Hz
- Dual-lead ECG: MLII (primary) and V5
- Annotations: Beat-level labels with sample indices
- Format: CSV files (`{patient_id}_ekg.csv`, `{patient_id}_annotations_1.csv`)

**Key Annotation Symbols:**
- **Normal**: N, L, R, e, j
- **Supraventricular (SV)**: A, a, J, S, n
- **Ventricular**: V, r, E, F
- **Other**: /, f, Q, ? (excluded from training)

---

## Proposed Changes

### Data Processing Pipeline

#### [NEW] [data_loader.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/data_loader.py)

**Purpose**: Load and parse MIT-BIH dataset

**Key Functions:**
- `load_ecg_record(patient_id)`: Load ECG signal and annotations for a patient
- `get_all_patient_ids()`: Return list of all available patient IDs
- `load_annotation_symbols()`: Load annotation symbol mapping

**Implementation Details:**
- Parse CSV files for ECG signals (MLII lead)
- Parse annotation files to extract beat indices and symbols
- Handle multiple annotation files per patient (e.g., [108_annotations_2.csv](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/MIT-BIH_Arrhythmia_Dataset/108_annotations_2.csv))

---

#### [NEW] [preprocessing.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/preprocessing.py)

**Purpose**: Signal cleaning and normalization

**Key Functions:**
- `bandpass_filter(signal, lowcut=0.5, highcut=40, fs=360)`: Apply Butterworth band-pass filter
- `remove_baseline_wander(signal)`: Remove low-frequency drift using high-pass filter
- `normalize_signal(signal)`: Z-score normalization per record
- `preprocess_record(signal)`: Complete preprocessing pipeline

**Implementation Details:**
- Use `scipy.signal.butter` for filter design
- Apply forward-backward filtering (`filtfilt`) to avoid phase distortion
- Preserve ECG morphology while removing noise

---

#### [NEW] [segmentation.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/segmentation.py)

**Purpose**: R-peak detection and beat segmentation

**Key Functions:**
- `pan_tompkins_detector(signal, fs=360)`: Implement Pan-Tompkins R-peak detection algorithm
- `extract_beat_windows(signal, r_peaks, window_size=360)`: Extract fixed-size windows around R-peaks
- `validate_r_peaks(signal, r_peaks, annotations)`: Compare detected vs annotated R-peaks

**Implementation Details:**
- Pan-Tompkins algorithm stages:
    1. Band-pass filtering (5-15 Hz)
    2. Derivative filter
    3. Squaring
    4. Moving window integration
    5. Adaptive thresholding
- Window extraction: 180 samples before R-peak, 180 samples after (total 360 samples ≈ 1 second)
- Align windows consistently for model input

---

#### [NEW] [label_engineering.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/label_engineering.py)

**Purpose**: Map MIT-BIH symbols to model labels

**Key Functions:**
- `map_symbol_to_class(symbol)`: Map annotation symbol to clinical class
- `create_stage1_labels(symbols)`: Binary labels (0=Normal, 1=Abnormal)
- `create_stage2_labels(symbols)`: Multi-class labels (0=SV, 1=Ventricular) for abnormal beats only

**Label Mapping:**

| MIT-BIH Symbol | Clinical Class | Stage 1 Label | Stage 2 Label |
|----------------|----------------|---------------|---------------|
| N, L, R, e, j  | Normal         | 0 (Normal)    | N/A           |
| A, a, J, S, n  | Supraventricular | 1 (Abnormal) | 0 (SV)        |
| V, r, E, F     | Ventricular    | 1 (Abnormal)  | 1 (Ventricular) |
| /, f, Q, ?     | Excluded       | Excluded      | Excluded      |

---

#### [NEW] [sqi_labeling.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/sqi_labeling.py)

**Purpose**: Generate Signal Quality Index labels

**Key Functions:**
- `calculate_snr(signal)`: Calculate signal-to-noise ratio
- `calculate_baseline_stability(signal)`: Measure baseline wander
- `calculate_kurtosis(signal)`: Measure signal distribution
- `label_signal_quality(signal)`: Assign GOOD/BAD label based on quality metrics

**Quality Criteria:**
- **GOOD signals**: SNR > 10 dB, baseline stability < 0.15, kurtosis 3-7
- **BAD signals**: SNR ≤ 10 dB, excessive baseline wander, or abnormal kurtosis

**Implementation Details:**
- Apply quality metrics to each beat window
- Use thresholds based on clinical standards
- Generate binary labels (0=BAD, 1=GOOD)

---

#### [NEW] [data_split.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/data_split.py)

**Purpose**: Inter-patient train-validation split

**Key Functions:**
- `split_patients(patient_ids, train_ratio=0.8, random_seed=42)`: Split by patient ID
- `get_train_val_data()`: Return train and validation datasets

**Implementation Details:**
- **Critical**: Split by patient ID to prevent data leakage
- Stratify by arrhythmia prevalence if possible
- Suggested split: 80% train (38 patients), 20% validation (10 patients)

---

#### [NEW] [augmentation.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/augmentation.py)

**Purpose**: ECG-specific data augmentation

**Key Functions:**
- `add_gaussian_noise(signal, std=0.05)`: Simulate sensor noise
- `add_baseline_wander(signal, amplitude=0.1, frequency=0.5)`: Simulate motion artifacts
- `amplitude_scaling(signal, scale_range=(0.8, 1.2))`: Random amplitude variation
- `time_stretch(signal, rate_range=(0.95, 1.05))`: Simulate heart rate variability
- `random_dropout(signal, dropout_prob=0.1)`: Simulate signal loss

**Implementation Details:**
- Apply augmentations randomly during training
- Preserve beat morphology (avoid over-augmentation)
- Use augmentation only on training set

---

### Model Architectures

#### [NEW] [models/sqi_model.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/models/sqi_model.py)

**Purpose**: Stage 0 - Signal Quality Index Model

**Architecture**: Ultra-lightweight 1D CNN

```
Input: (360, 1) - 1-second ECG window

Block 1 (Initial features):
  - Conv1D(16 filters, kernel=5, padding=same)
  - BatchNormalization
  - ReLU activation
  - MaxPooling1D(pool_size=2)
  - Dropout(0.2)

Block 2 (Mid-level features):
  - Conv1D(32 filters, kernel=3, padding=same)
  - BatchNormalization
  - ReLU activation
  - MaxPooling1D(pool_size=2)
  - Dropout(0.3)

Block 3 (High-level features):
  - Conv1D(64 filters, kernel=3, padding=same)
  - BatchNormalization
  - ReLU activation
  - GlobalAveragePooling1D

Classification Head:
  - Dense(16, activation=relu)
  - Dropout(0.3)
  - Dense(1, activation=sigmoid)

Output: Quality score (0=BAD, 1=GOOD)
```

**Design Rationale:**
- Minimal parameter count (~15K) for fastest inference
- Simple architecture for binary quality assessment
- Serves as pre-filter before arrhythmia detection
- Sigmoid output for quality probability

---

#### [NEW] [models/stage1_model.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/models/stage1_model.py)

**Purpose**: Stage 1 - Abnormality Screening Model

**Architecture**: Lightweight Multi-Scale 1D CNN

```
Input: (360, 1) - 1-second ECG window

Block 1 (Fine-grained features):
  - Conv1D(32 filters, kernel=3, activation=relu)
  - BatchNormalization
  - MaxPooling1D(pool_size=2)
  - Dropout(0.2)

Block 2 (Medium-scale features):
  - Conv1D(64 filters, kernel=5, activation=relu)
  - BatchNormalization
  - MaxPooling1D(pool_size=2)
  - Dropout(0.3)

Block 3 (Coarse features):
  - Conv1D(128 filters, kernel=7, activation=relu)
  - BatchNormalization
  - MaxPooling1D(pool_size=2)
  - Dropout(0.3)

Global Feature Extraction:
  - GlobalAveragePooling1D

Classification Head:
  - Dense(64, activation=relu)
  - Dropout(0.4)
  - Dense(1, activation=sigmoid)

Output: Binary probability (Normal vs Abnormal)
```

**Design Rationale:**
- Multi-scale kernels (3, 5, 7) capture different temporal patterns
- Small parameter count (~50K) for edge deployment
- High dropout for regularization
- Sigmoid output for binary classification

---

#### [NEW] [models/stage2_model.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/models/stage2_model.py)

**Purpose**: Stage 2 - Arrhythmia Diagnosis Model

**Architecture**: Multi-Scale CNN with Dilated Convolutions

```
Input: (360, 1) - 1-second ECG window (abnormal beats only)

Block 1 (Local features):
  - Conv1D(64 filters, kernel=3, dilation=1, activation=relu)
  - BatchNormalization
  - MaxPooling1D(pool_size=2)
  - Dropout(0.2)

Block 2 (Medium-range features):
  - Conv1D(128 filters, kernel=5, dilation=2, activation=relu)
  - BatchNormalization
  - MaxPooling1D(pool_size=2)
  - Dropout(0.3)

Block 3 (Long-range features):
  - Conv1D(256 filters, kernel=7, dilation=4, activation=relu)
  - BatchNormalization
  - MaxPooling1D(pool_size=2)
  - Dropout(0.3)

Block 4 (Contextual features):
  - Conv1D(256 filters, kernel=3, dilation=8, activation=relu)
  - BatchNormalization
  - GlobalAveragePooling1D

Classification Head:
  - Dense(128, activation=relu)
  - Dropout(0.5)
  - Dense(64, activation=relu)
  - Dropout(0.4)
  - Dense(2, activation=softmax)

Output: Class probabilities (SV vs Ventricular)
```

**Design Rationale:**
- Dilated convolutions increase receptive field without pooling
- Larger model (~150K parameters) for complex arrhythmia patterns
- Deeper network for fine-grained classification
- Softmax output for multi-class classification

---

### Training Pipeline

#### [NEW] [train_sqi.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/train_sqi.py)

**Purpose**: Train SQI signal quality model

**Training Configuration:**
- **Loss**: Binary cross-entropy
- **Optimizer**: Adam (lr=0.001)
- **Batch size**: 64
- **Epochs**: 50 (with early stopping)
- **Callbacks**:
    - EarlyStopping (patience=10, monitor='val_loss')
    - ReduceLROnPlateau (patience=5, factor=0.5)
    - ModelCheckpoint (save best model based on val_accuracy)

**Key Metrics:**
- **Primary**: Accuracy
- **Secondary**: Precision, Recall, AUC

**Implementation Details:**
- Use quality-labeled data from `sqi_labeling.py`
- Balance GOOD and BAD quality samples
- Save training history for analysis

---

#### [NEW] [train_stage1.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/train_stage1.py)

**Purpose**: Train Stage 1 abnormality screening model

**Training Configuration:**
- **Loss**: Binary cross-entropy
- **Optimizer**: Adam (lr=0.001)
- **Batch size**: 64
- **Epochs**: 100 (with early stopping)
- **Class weighting**: Compute from class distribution to handle imbalance
- **Callbacks**:
    - EarlyStopping (patience=15, monitor='val_loss')
    - ReduceLROnPlateau (patience=7, factor=0.5)
    - ModelCheckpoint (save best model based on val_recall)

**Key Metrics:**
- **Primary**: Sensitivity (Recall) - minimize false negatives
- **Secondary**: Specificity, F1-score, AUC-ROC

**Implementation Details:**
- Use data generator with augmentation
- Prioritize sensitivity over precision (screening model)
- Save training history for analysis

---

#### [NEW] [train_stage2.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/train_stage2.py)

**Purpose**: Train Stage 2 arrhythmia diagnosis model

**Training Configuration:**
- **Loss**: Categorical cross-entropy
- **Optimizer**: Adam (lr=0.0005)
- **Batch size**: 32
- **Epochs**: 150 (with early stopping)
- **Class balancing**: Oversample minority class or use class weights
- **Callbacks**:
    - EarlyStopping (patience=20, monitor='val_loss')
    - ReduceLROnPlateau (patience=10, factor=0.5)
    - ModelCheckpoint (save best model based on val_f1_score)

**Key Metrics:**
- **Primary**: F1-score (balanced precision and recall)
- **Secondary**: Per-class precision, recall, confusion matrix

**Implementation Details:**
- Train only on abnormal beats (filtered from Stage 1)
- Balance SV and Ventricular classes
- Strong regularization to prevent overfitting

---

### Evaluation & Analysis

#### [NEW] [evaluate.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/evaluate.py)

**Purpose**: Comprehensive model evaluation

**Key Functions:**
- `evaluate_sqi(model, val_data)`: SQI metrics
- `evaluate_stage1(model, val_data)`: Stage 1 metrics
- `evaluate_stage2(model, val_data)`: Stage 2 metrics
- `plot_confusion_matrix(y_true, y_pred, labels)`: Visualize confusion matrix
- `plot_roc_curve(y_true, y_scores)`: ROC curve for binary classifiers
- `analyze_misclassifications(model, val_data)`: Identify failure cases

**Medical Metrics:**

**SQI (Quality Assessment):**
- **Accuracy**: Overall quality classification accuracy
- **Precision**: Proportion of predicted GOOD signals that are truly good
- **Recall**: Proportion of actual GOOD signals correctly identified
- **AUC**: Overall discriminative ability

**Stage 1 (Abnormality Screening):**
- **Sensitivity (Recall)**: TP / (TP + FN) - critical for screening
- **Specificity**: TN / (TN + FP)
- **F1-score**: Harmonic mean of precision and recall
- **AUC-ROC**: Overall discriminative ability
- **Miss rate**: FN / (TP + FN) - must be minimized

**Stage 2 (Arrhythmia Diagnosis):**
- **Per-class Precision**: TP / (TP + FP) for SV and Ventricular
- **Per-class Recall**: TP / (TP + FN) for SV and Ventricular
- **F1-score**: Per-class and macro-averaged
- **Confusion Matrix**: Visualize SV vs Ventricular confusion

---

#### [NEW] [visualize.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/visualize.py)

**Purpose**: Visualization utilities

**Key Functions:**
- `plot_raw_vs_filtered(raw_signal, filtered_signal)`: Compare preprocessing
- `plot_r_peak_detection(signal, r_peaks, annotations)`: Validate R-peak detection
- `plot_beat_samples(beats, labels)`: Visualize beat windows by class
- `plot_training_history(history)`: Training curves (loss, accuracy, etc.)
- `plot_class_distribution(labels)`: Dataset balance analysis
- `plot_quality_distribution(quality_scores)`: SQI score distribution

---

### Edge AI Optimization

#### [NEW] [convert_to_tflite.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/convert_to_tflite.py)

**Purpose**: Convert Keras models to TensorFlow Lite

**Key Functions:**
- `convert_model(keras_model, output_path, quantization='dynamic')`: Convert with optional quantization
- `benchmark_tflite_model(tflite_path, test_data)`: Measure inference time and accuracy
- `compare_models(keras_model, tflite_model, test_data)`: Compare accuracy before/after conversion

**Quantization Strategies:**

1. **Dynamic Range Quantization** (Default):
    - Post-training quantization
    - Weights: INT8, Activations: FLOAT32
    - ~4x size reduction, minimal accuracy loss
    - No representative dataset required

2. **Full Integer Quantization** (Optional):
    - Weights and activations: INT8
    - ~4x size reduction, ~2-4x speedup
    - Requires representative dataset
    - Slight accuracy degradation (acceptable if <2%)

**Implementation Details:**
- Use `tf.lite.TFLiteConverter.from_keras_model()`
- Apply optimizations: `tf.lite.Optimize.DEFAULT`
- Validate accuracy on validation set
- Measure model size and inference latency

---

#### [NEW] [inference.py](file:///c:/Users/Shreyas/Downloads/RhythmAI-MLModeTrainging/src/inference.py)

**Purpose**: Three-stage inference pipeline

**Key Functions:**
- `predict_three_stage(ecg_window, sqi_model, stage1_model, stage2_model)`: Complete inference
- `load_tflite_model(model_path)`: Load TFLite model
- `run_tflite_inference(interpreter, input_data)`: Run TFLite inference

**Inference Flow:**
1. Preprocess ECG window (filter, normalize)
2. Run SQI model:
    - If BAD quality (score < threshold): Return "BAD_QUALITY"
    - If GOOD quality (score ≥ threshold): Proceed to Stage 1
3. Run Stage 1 model:
    - If Normal (prob < threshold): Return "Normal"
    - If Abnormal (prob ≥ threshold): Proceed to Stage 2
4. Run Stage 2 model:
    - Return "Supraventricular" or "Ventricular"

**Threshold Tuning:**
- SQI threshold: 0.5 (default) - adjustable for quality sensitivity
- Stage 1 threshold: 0.5 (default) - can lower to 0.3 for higher sensitivity
- Stage 2: Argmax of softmax probabilities

---

### Project Structure

```
RhythmAI-MLModeTrainging/
├── MIT-BIH_Arrhythmia_Dataset/
│   ├── {patient_id}_ekg.csv
│   ├── {patient_id}_annotations_1.csv
│   └── annotation_symbols.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── label_engineering.py
│   ├── sqi_labeling.py
│   ├── data_split.py
│   ├── augmentation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sqi_model.py
│   │   ├── stage1_model.py
│   │   └── stage2_model.py
│   ├── train_sqi.py
│   ├── train_stage1.py
│   ├── train_stage2.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── convert_to_tflite.py
│   └── inference.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_edge_optimization.ipynb
├── models/
│   ├── sqi_model.h5
│   ├── sqi_model.tflite
│   ├── stage1_model.h5
│   ├── stage1_model.tflite
│   ├── stage2_model.h5
│   └── stage2_model.tflite
├── results/
│   ├── training_history/
│   ├── evaluation_metrics/
│   └── visualizations/
├── requirements.txt
└── README.md
```

---

## Verification Plan

### Automated Tests

#### 1. Data Pipeline Validation

**Test Script**: `tests/test_data_pipeline.py`

**Test Cases:**
- Load all 48 patient records successfully
- Verify ECG signal shape (650000 samples at 360 Hz)
- Validate annotation parsing (indices and symbols)
- Check preprocessing output (filtered signal has same length)
- Verify R-peak detection (compare with annotations, tolerance ±10 samples)
- Validate window extraction (all windows have shape (360, 1))

**Run Command:**
```bash
python -m pytest tests/test_data_pipeline.py -v
```

---

#### 2. Label Engineering Validation

**Test Script**: `tests/test_label_engineering.py`

**Test Cases:**
- Verify symbol-to-class mapping for all annotation types
- Check Stage 1 label distribution (Normal vs Abnormal)
- Check Stage 2 label distribution (SV vs Ventricular)
- Validate SQI label generation (GOOD vs BAD)
- Ensure excluded symbols (/, f, Q, ?) are filtered out
- Validate inter-patient split (no patient overlap between train/val)

**Run Command:**
```bash
python -m pytest tests/test_label_engineering.py -v
```

---

#### 3. Model Architecture Validation

**Test Script**: `tests/test_models.py`

**Test Cases:**
- SQI model input shape: (None, 360, 1)
- SQI model output shape: (None, 1)
- Stage 1 model input shape: (None, 360, 1)
- Stage 1 model output shape: (None, 1)
- Stage 2 model input shape: (None, 360, 1)
- Stage 2 model output shape: (None, 2)
- Verify parameter counts (SQI < 20K, Stage 1 < 100K, Stage 2 < 200K)
- Test forward pass with dummy data

**Run Command:**
```bash
python -m pytest tests/test_models.py -v
```

---

### Training Validation

#### 4. SQI Model Training

**Script**: `src/train_sqi.py`

**Success Criteria:**
- Training completes without errors
- Validation accuracy ≥ 90%
- Validation AUC ≥ 0.95
- Model converges (loss decreases over epochs)
- No severe overfitting (train-val gap < 10%)

**Run Command:**
```bash
python src/train_sqi.py --dataset_path ../MIT-BIH_Arrhythmia_Dataset --epochs 50 --batch_size 64
```

**Expected Output:**
- Saved model: `models/sqi_model.h5`
- Training history: `results/sqi_training_history.png`
- Validation metrics printed to console

---

#### 5. Stage 1 Model Training

**Script**: `src/train_stage1.py`

**Success Criteria:**
- Training completes without errors
- Validation sensitivity ≥ 95% (high recall for screening)
- Validation specificity ≥ 85%
- Model converges (loss decreases over epochs)
- No severe overfitting (train-val gap < 10%)

**Run Command:**
```bash
python src/train_stage1.py --dataset_path ../MIT-BIH_Arrhythmia_Dataset --epochs 100 --batch_size 64
```

**Expected Output:**
- Saved model: `models/stage1_model.h5`
- Training history: `results/training_history/stage1_history.json`
- Validation metrics printed to console

---

#### 6. Stage 2 Model Training

**Script**: `src/train_stage2.py`

**Success Criteria:**
- Training completes without errors
- Validation F1-score ≥ 90% (balanced precision/recall)
- Per-class recall ≥ 85% for both SV and Ventricular
- Confusion matrix shows low inter-class confusion (<15%)

**Run Command:**
```bash
python src/train_stage2.py --dataset_path ../MIT-BIH_Arrhythmia_Dataset --epochs 150 --batch_size 32
```

**Expected Output:**
- Saved model: `models/stage2_model.h5`
- Training history: `results/training_history/stage2_history.json`
- Confusion matrix: `results/evaluation_metrics/stage2_confusion_matrix.png`

---

### Evaluation Validation

#### 7. Comprehensive Model Evaluation

**Script**: `src/evaluate.py`

**Metrics to Validate:**

**SQI:**
- Accuracy ≥ 90%
- Precision ≥ 88%
- Recall ≥ 88%
- AUC ≥ 0.95

**Stage 1:**
- Sensitivity ≥ 95%
- Specificity ≥ 85%
- F1-score ≥ 90%
- AUC-ROC ≥ 0.95

**Stage 2:**
- Macro F1-score ≥ 90%
- SV Precision ≥ 88%, Recall ≥ 88%
- Ventricular Precision ≥ 88%, Recall ≥ 88%

**Run Command:**
```bash
python src/evaluate.py --sqi_model models/sqi_model.h5 --stage1_model models/stage1_model.h5 --stage2_model models/stage2_model.h5
```

**Expected Output:**
- Detailed metrics report: `results/evaluation_metrics/evaluation_report.txt`
- ROC curves: `results/evaluation_metrics/roc_curves.png`
- Confusion matrices: `results/evaluation_metrics/confusion_matrices.png`

---

### Edge Optimization Validation

#### 8. TensorFlow Lite Conversion

**Script**: `src/convert_to_tflite.py`

**Success Criteria:**
- Models convert successfully to TFLite format
- Dynamic quantization reduces model size by ~90%
- Accuracy degradation < 2% compared to Keras models
- Inference time < 15ms per beat on CPU (all 3 stages)

**Run Command:**
```bash
python src/convert_to_tflite.py --convert_both --dataset_path ../MIT-BIH_Arrhythmia_Dataset
```

**Expected Output:**
- TFLite models: `models/sqi_model.tflite`, `models/stage1_model.tflite`, `models/stage2_model.tflite`
- Benchmark report: `results/edge_optimization/tflite_benchmark.txt`
- Size comparison: Total Keras (~7.4MB) → Total TFLite (~654KB)

---

#### 9. End-to-End Inference Test

**Script**: `src/inference.py`

**Test Cases:**
- Load TFLite models successfully
- Run three-stage inference on validation set
- Verify output format (class label + confidence)
- Measure average inference time per beat

**Run Command:**
```bash
python src/inference.py --test_samples 1000 --use_tflite
```

**Expected Output:**
- Inference results: `results/inference_results.json`
- Average inference time: < 15ms per beat (all 3 stages)
- Classification accuracy matches evaluation metrics

---

### Visualization Validation

#### 10. Data Exploration Notebook

**Notebook**: `notebooks/01_data_exploration.ipynb`

**Validation Steps:**
1. Load and visualize raw ECG signals from 5 patients
2. Display annotation distribution across all patients
3. Plot class imbalance (Normal vs SV vs Ventricular)
4. Visualize beat-to-beat variability
5. Show signal quality distribution

**Run Command:**
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

#### 11. Preprocessing Validation Notebook

**Notebook**: `notebooks/02_preprocessing_validation.ipynb`

**Validation Steps:**
1. Compare raw vs filtered signals (band-pass filter)
2. Visualize R-peak detection on 10 sample beats
3. Display extracted beat windows (aligned)
4. Show augmentation effects (noise, baseline wander, scaling)
5. Demonstrate quality assessment on sample beats

**Run Command:**
```bash
jupyter notebook notebooks/02_preprocessing_validation.ipynb
```

---

## Dependencies

**Requirements** (`requirements.txt`):
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.10.0
jupyter>=1.0.0
pytest>=7.0.0
```

**Installation:**
```bash
pip install -r requirements.txt
```

---

## Success Metrics Summary

| Metric | SQI Target | Stage 1 Target | Stage 2 Target |
|--------|------------|----------------|----------------|
| Accuracy/F1 | ≥ 90% | ≥ 90% | ≥ 90% |
| Sensitivity | N/A | ≥ 95% | N/A |
| Specificity | N/A | ≥ 85% | N/A |
| Per-Class Recall | N/A | N/A | ≥ 85% (both classes) |
| Model Size (TFLite) | < 20 KB | < 100 KB | < 600 KB |
| Inference Time | < 2 ms | < 5 ms | < 10 ms |
| Accuracy Degradation (TFLite) | < 2% | < 2% | < 2% |

---
