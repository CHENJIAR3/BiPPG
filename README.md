<div align="center">

# BiPPG

**Large-scale bilateral cardiovascular monitoring via wearable rings**

<p align="center">

</p>

> A multi-task bilateral learning framework, which is **better** than unilateral configurations.

**First Author:** Jiarong Chen (E-mail: jiarong.chen@sjtu.edu.cn)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Statistical Analysis](#statistical-analysis)
- [Citation](#citation)
- [License](#license)

---

## Overview

**BiPPG** is a research codebase designed for the systematic study of cardiovascular monitoring using photoplethysmography (PPG) signals acquired from bilateral measurement sites (e.g., left and right fingers). 
The framework provides:

- **End-to-end pipelines** from raw PPG preprocessing to clinical metric reporting
- **Multi-model benchmarking** across multiple deep learning architectures spanning CNNs, RNNs, and Transformers
- **Subpopulation-aware evaluation** stratified by gender, BMI, age, BP level, and measurement position
- **Statistical validation** including normality testing, non-parametric comparisons, and subject-level analysis
- **Bilateral consistency filtering** to enforce bilateral consistency before model evaluation

---

## Project Structure

```
BiPPG/
│
├── dataloaders/                    # Data ingestion and augmentation
│   ├── dataloading.py              # Dataset class, pt loading
│   └── transform.py               # Signal normalization, windowing, augmentation transforms
│
├── evaluation/                     # Inference, metrics, and statistical analysis
│   ├── get_prediction.py           # Run inference
│   ├── load_prediction.py          # Load prediction results 
│   ├── model_metric.py             # MAE, R, Pearson correlation 
│   ├── baseline_hr_nk.py           # Heart-rate baseline using neurokit2
│   ├── Bilateral_stats.py          # Sample-level
│   ├── Bilateral_stats_subject_level.py   # Subject-level
│   ├── predictions_to_excel.py     # Export predictions and metrics to formatted Excel
│   ├── subpopulation_analysis.py           # Subgroup stratification 
│   ├── subpopulation_analysis_BD.py        # Subpopulation BD analysis
│   └── subpopulation_analysis_BD_subject_level.py  # Subject-level BD Subpopulatio analysis
│
├── model_training/
│   └── train_baseline.py           # Training loop, LR scheduling, checkpointing
│
├── models/                         # Deep learning architecture zoo
│   ├── model_loading.py            # Unified model factory / registry
│   ├── ResNet1D.py                
│   ├── net1D.py                    
│   ├── ACNN.py                     
│   ├── CRNN.py                     
│   ├── CSFM.py                     
│   ├── LSTM.py                   
│   ├── efficientnet.py           
│   ├── papagei_resnet.py           
│   ├── PatchTST.py                
│   ├── Informer.py              
│   ├── Autoformer.py               
│   └── iTransformers.py            
│
└── Preprocessing/                  # Offline preprocessing and dataset curation
    ├── get_information.py          # Extract metadata (subject info, BP reference, position)
    ├── analysis_subgroup.py        # Subgroup membership assignment and QC filtering
    └── data_splits_paths.pkl       # Pre-computed train/val/test split file paths, inter-subject
```


## Installation

### Prerequisites

- Python ≥ 3.8
- CUDA ≥ 11.3 (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/CHENJIAR3/BiPPG.git
cd BiPPG

# Create and activate a virtual environment
conda env create -f environment.yml


## Data Preparation

BiPPG expects preprocessed data stored as `.pkl` files with the following convention:

```
/your/dataset/root/
├── <subject_id_1>.pkl
├── <subject_id_2>.pkl
└── ...
```

Each `.pkl` file is a `pd.DataFrame` with (at minimum) the columns:

| Column | Description |
|---|---|
| `ppg_idx` | PPG segment index within the session |
| `bp_idx` | Blood pressure measurement index |
| `sbp_fix` | Reference systolic BP (mmHg) |
| `dbp_fix` | Reference diastolic BP (mmHg) |
| `pr_ref` | Reference pulse rate (bpm) |
| `gender` | Participant gender (1 for male, and 2 for female) |
| `age` | Participant age (years) |
| `bmi` | Body mass index (kg/m²) |
| `position` | Measurement position |
| `ppg_*` | PPG signal columns  |

---

## Usage

### 1. Train a Model
### 2. Run Inference and Get Prediction
### 3. Compute Metrics
### 4. Subpopulation Analysis
### 5. Statistical Comparison (All vs. Filtered Cohort)


## Evaluation

BiPPG reports the following metrics:

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| Pearson R | Linear correlation with reference (Correlation analysis) |
| ME ± SD | Mean Error ± Standard Deviation (Bland-Altman) |

---

## Statistical Analysis

The `evaluation/Bilateral_stats.py` module implements a rigorous two-group comparison pipeline between the full cohort (`results_df`) and the quality-filtered cohort (`results_used`):
### Continuous Variables — `bmi`, `age`, `sbp_fix`, `dbp_fix`, `pr_ref`
1. **Normality testing** — Shapiro-Wilk (n ≤ 5,000) or Kolmogorov-Smirnov (n > 5,000)
2. Both groups normal → **Welch's independent-samples t-test**
3. Either group non-normal → **Mann-Whitney U test**
### Categorical Variables — `gender`
- **Pearson's χ² test** on the contingency table
Significance level: **α = 0.05**

---

## Citation

If you use BiPPG in your research, please cite:

```bibtex
@article{
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Developed by **Jiarong Chen** &nbsp;|&nbsp; Only for academic and research use

</div>
