# ADNI Multimodal Diagnostic Pipeline

This project implements a machine learning pipeline to classify Alzheimer’s Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) participants using multimodal baseline data from the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)**. The pipeline integrates **clinical/cognitive/demographic measures, MRI volumetric features, and CSF biomarkers**.

---

## 📂 Project Structure

```
├── data/            # Raw and processed datasets, plus plots
├── pipelines/       # Jupyter notebooks for modality-specific and multimodal pipelines
├── output/          # Model results, metrics, and plots
├── Main_ADNI_Orchestrator.ipynb   # Master workflow notebook
```

---

##  Overview

- **Clinical/Cognitive/Demographic Pipeline** – preprocesses cognitive tests, demographics, and clinical data  
- **MRI Pipeline** – handles volumetric MRI features with QC and feature selection  
- **CSF Pipeline** – prepares biomarker features (Tau, pTau, Aβ42, etc.)  
- **Data Fusion Pipeline** – merges modalities into a unified feature matrix  
- **Multimodal Diagnostic Pipeline** – trains ML models and evaluates performance (with & without CSF)  

---

##  Outputs

- Metrics tables (CV Accuracy, Macro-F1, ROC-AUC)  
- ROC curves, feature importance plots, and correlation heatmaps  
- Fusion dataset for multimodal modelling  

---

## ⚙️ Requirements

- Python 3.9+  
- Jupyter Notebook  
- Core libraries: pandas, numpy, scikit-learn, xgboost, imbalanced-learn, matplotlib, seaborn, shap  

Install dependencies:  
```bash
pip install -r requirements.txt
```

---

##  How to Run

1. Place raw ADNI datasets into `data/raw`  
2. Run preprocessing pipelines in `pipelines/`  
3. Use the Data Fusion Pipeline to create a combined dataset  
4. Run multimodal diagnostic notebooks to train and evaluate models  
5. Results are saved in `output/`  

---

##  Ethics & Data Use

- Data obtained from **ADNI** under a Data Use Agreement (DUA)  
- Approved by **University of Nottingham SOP 2.1** ethics protocol  
- See *Appendix A* in dissertation for SOP and DMP details  

---

##  Citation

If you use this work, please cite ADNI:  
> Alzheimer’s Disease Neuroimaging Initiative (ADNI). http://adni.loni.usc.edu  
