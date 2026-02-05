# Heart Disease Prediction (Streamlit)

A Streamlit web app that predicts the likelihood of heart disease from patient parameters.
Supports multiple ML pipelines (AdaBoost, RandomForest, GradientBoosting, XGBoost) saved as `.joblib`.

## Features
- Input patient parameters in sidebar form
- Load saved ML pipelines from `models/`
- Auto-pick best model or manual model selection
- Show prediction probability (if supported by the model)

## Project Structure
- `deploy.py` : Streamlit app
- `models/` : saved pipelines (`*.joblib`)
- `notebooks/` : training notebooks
- `requirements.txt` : dependencies

## How to Run Locally
1. Install dependencies:
```bash
pip install -r requirements.txt
