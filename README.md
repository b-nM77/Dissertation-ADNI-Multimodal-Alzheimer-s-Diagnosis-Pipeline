<!-- PROJECT BANNER -->
<h1 align="center">🧠 ADNI Multimodal Alzheimer’s Diagnosis Pipeline</h1>
<p align="center">
  <em>AI-based multimodal machine learning pipeline for baseline diagnostic classification (CN vs MCI vs AD)</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange" />
  <img src="https://img.shields.io/badge/Scikit--learn-ML%20Pipeline-green" />
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-yellow" />
</p>

---

##  Project Overview

This repository contains the complete implementation of my Master’s dissertation:

### **“An AI-Based Multimodal Pipeline for Diagnostic Classification of Alzheimer’s Disease Using Clinical, MRI, and CSF Biomarkers from ADNI.”**

The project builds a full **end-to-end multimodal ML system** to classify:

- **CN — Cognitively Normal**
- **MCI — Mild Cognitive Impairment**
- **AD — Alzheimer’s Disease**
  
Using baseline ADNI data from participants who carry the **APOE-ε4 allele — a major genetic risk factor for Alzheimer’s Disease** — the pipeline integrates:

-  Clinical, Cognitive, Demographic features  
-  CSF Biomarkers (Aβ42, Tau, pTau + ratios)  
-  MRI Volumetric FreeSurfer ROI features (~300 → filtered to ~100)  

---

## Multimodal Pipeline Architecture

### **1. Modality-Specific Pipelines**
- **Clinical/Cognitive:** MICE imputation, IQR-based outlier clipping, One-hot encoding for categorical variables, Standardisation  
- **CSF Biomarkers:**
   - Raw biomarkers: Aβ42, Tau, pTau
   - Derived ratios: Tau/Aβ42, pTau/Aβ42
   - IQR clipping to enforce biological plausibility 
- **MRI Volumes:**
  - ADNI FreeSurfer volumetric ROIs (~300)
  - Built-in QC filtering
  - Unsupervised ROI filtering:
      - Missingness > 30% removal
      - Near-zero variance removal
      - High correlation pruning (|ρ| ≥ 0.9)
  - Supervised ANOVA F-test for discriminative ROI selection
  - Final compact set of ~100 clinically meaningful ROIs

  
### **2. Data Fusion Pipeline**
All cleaned modality outputs are merged on PTID → producing a unified feature vector per participant.

Two fusion datasets:
1. **Full Multimodal Fusion (Clinical + MRI + CSF)**
2. **Multimodal Without CSF** (ablation study)

### **3. Multimodal Diagnostic Modelling**
Models compared:
- Support Vector Machine (RBF)
- Random Forest
- XGBoost
- **Stacking Ensemble (best)**

Evaluation strategy:
 - **Stratified 5-fold CV**
 - **20% independent test set**
 - **ADASYN for class imbalance**
 - **Metrics:** Accuracy, Macro-F1, Macro ROC-AUC, Confusion Matrix 

---

##  Results Summary

### **Multimodal Pipeline (All Modalities)**

| Model               | Test Accuracy | Macro-F1 | ROC-AUC (OvR, Macro) |
|---------------------|--------------:|---------:|----------------------:|
| **Stacking Ensemble** | **76.5%**     | **0.761** | **0.900** |
| XGBoost             | 75.8%         | 0.761    | 0.896 |
| Random Forest       | 75.2%         | 0.752    | 0.893 |
| SVM (RBF)           | 70.5%         | 0.710    | 0.875 |

### **Multimodal Pipeline (Excluding CSF Biomarkers)**

| Model               | Test Accuracy | Macro-F1 | ROC-AUC (OvR, Macro) |
|---------------------|--------------:|---------:|----------------------:|
| **Stacking Ensemble** | **74.5%**     | **0.740** | **0.888** |
| Random Forest       | 74.5%         | 0.748    | 0.883 |
| XGBoost             | 73.8%         | 0.737    | 0.883 |
| SVM (RBF)           | 70.5%         | 0.709    | 0.868 |


**Key Insight:**  
 - **CSF biomarkers provide a consistent performance boost.**  
 - **Stacking Ensemble generalises best across both pipelines.**

---

##  Project Structure

```bash
ADNI-Multimodal-Pipeline/
│
├── data/
│   ├── raw/                          # Raw ADNI files (not uploaded)
│   ├── processed/                    # Cleaned per-modality outputs
│   └── fusion/                       # Final fusion datasets
│
├── pipelines/
│   ├── clinical_cognitive_demographic_pipeline.ipynb
│   ├── MRI_pipeline.ipynb
│   ├── CSF_biomarker_pipeline.ipynb
│   ├── Data_Fusion_Pipeline.ipynb
│   ├── Multimodal_Diagnostic_Pipeline_All_Modalities.ipynb
│   └── Multimodal_Diagnostic_Pipeline_Excluding_CSF_Biomarkers.ipynb
│
├── output/
│   ├── metrics/                      # Accuracy, F1, ROC-AUC tables
│   ├── plots/                        # ROC curves, heatmaps, feature importance
│   └── models/                       # Saved models (optional)
│
├── Main_ADNI_Orchestrator.ipynb      # Full end-to-end execution notebook
└── README.md

```

##  Installation & Requirements
#### Install Dependencies
```bash
pip install -r requirements.txt
```
#### Required Libraries
 - pandas, numpy, scikit-learn, xgboost, imbalanced-learn, matplotlib, seaborn.

## How to Run the Pipeline
 - Download ADNI baseline data (requires ADNI account + DUA)
 - Place raw files into:
     ```bash
     data/raw/
     ```
 - Execution:
   ```bash
    **Recommended:** Single entry point (orchestrator)
     1. Open `Main_ADNI_Orchestrator.ipynb`.
     2. Run all cells from top to bottom.

   This notebook acts as the full end-to-end execution script and sequentially runs the underlying modality pipelines:

    - `clinical_cognitive_pipeline.ipynb`
    - `CSF_biomarker_pipeline.ipynb`
    - `MRI_pipeline.ipynb`
    - `Data Fusion Pipeline.ipynb`
    - `Multimodal Diagnostic Pipeline (All Modalities).ipynb`
    - `Multimodal Diagnostic Pipeline Excluding CSF Biomarkers.ipynb`

   It will:
    - Load raw ADNI data from `data/raw/`
    - Generate processed per-modality outputs in `data/processed/`
    - Build fusion datasets in `data/fusion/`
    - Train and evaluate multimodal models
    - Save metrics and plots into `output/`

   Optional: Run step-by-step (for exploration) : If you prefer to explore each stage manually, run the notebooks in this order:

    1. `clinical_cognitive_pipeline.ipynb`
    2. `CSF_biomarker_pipeline.ipynb`
    3. `MRI_pipeline.ipynb`
    4. `Data Fusion Pipeline.ipynb`
    5. `Multimodal Diagnostic Pipeline (All Modalities).ipynb`
    6. `Multimodal Diagnostic Pipeline Excluding CSF Biomarkers.ipynb`

   Results are saved under output/ (metrics, plots, ROC curves)
   ```

## Ethics & Data Use
 - Data accessed under ADNI Data Use Agreement (DUA)
 - Approved under University of Nottingham SOP 2.1 (Ethics)
 - ADNI provides pseudonymised participant IDs
 - Raw data cannot be uploaded or redistributed

This repo contains only processing logic, metadata, and derivative outputs

## Citation

If you use this work, please cite ADNI:

Alzheimer’s Disease Neuroimaging Initiative (ADNI)
http://adni.loni.usc.edu/

## Author
 - Madhura Besagarahalli Nagaraju
 - MSc Computer Science (Artificial Intelligence)
 - University of Nottingham (2025 Cohort)
