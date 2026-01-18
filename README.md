# NHANES Biological Age Predictor with XAI

This repository contains a clinical decision support system for predicting biological age using NHANES data, featuring Explainable AI (XAI) insights.

## 📂 Repository Structure

| File | Description |
|------|-------------|
| **`dashboard.py`** | **[Main Application]** The Streamlit dashboard for deployment. Contains the full "Glass-Box" interface with SHAP waterfall plots and DiCE counterfactuals. |
| **`xai-age-predictor-script.py`** | **[Analysis Script]** The original research script used for data analysis, model training experiments, and generating pre-computed limits. |
| **`input/`** | Contains the NHANES dataset (`NHANES_age_prediction.csv`). |

## 🚀 How to Run the Dashboard

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Streamlit:**
   ```bash
   streamlit run dashboard.py
   ```

3. **View in Browser:**
   Open http://localhost:8501

## 🛠 Features

- **XGBoost Classification:** Predicts age group (Child, Adult, Aged).
- **SHAP (Diagnostic Insight):** Explains *why* a prediction was made.
- **DiCE (Clinical Recourse):** Suggests *actionable* changes to improve biological age.
- **Stakeholder Views:** Tailored summaries for Clinicians, Patients, and Policymakers.

## ☁️ Deployment

This project is ready for Streamlit Cloud.
- **Entry point:** `dashboard.py`
- **Dependencies:** `requirements.txt`
