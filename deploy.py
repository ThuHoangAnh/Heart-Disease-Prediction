# main.py
import os
import glob
import joblib
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# -----------------------------
# UI Styling (Background + Sidebar + Buttons)
# -----------------------------
st.markdown("""
<style>
/* Force background everywhere */
html, body, [class*="css"]  {
  background: transparent !important;
}

/* App background */
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px circle at 10% 10%, rgba(255, 0, 128, 0.16), transparent 45%),
              radial-gradient(900px circle at 90% 20%, rgba(0, 180, 255, 0.14), transparent 50%),
              linear-gradient(135deg, #0b1020 0%, #070a12 55%, #0b0f1e 100%) !important;
}

/* Top header bar */
[data-testid="stHeader"] {
  background: rgba(0,0,0,0) !important;
}

/* Sidebar */
[data-testid="stSidebar"] > div:first-child {
  background: rgba(16, 18, 27, 0.78) !important;
  backdrop-filter: blur(10px);
  border-right: 1px solid rgba(255,255,255,0.08);
}

/* Make main content a “glass” panel */
[data-testid="stMainBlockContainer"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 22px;
  padding: 24px 24px;
  backdrop-filter: blur(10px);
}

/* Buttons */
.stButton > button {
  border-radius: 12px;
  padding: 0.6rem 1rem;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.07);
}
.stButton > button:hover {
  border-color: rgba(255,255,255,0.32);
  background: rgba(255,255,255,0.12);
}
</style>
""", unsafe_allow_html=True)

st.title("💗 Heart Disease Prediction App")
st.write("Enter patient data. The model will predict the likelihood of heart disease.")

# -----------------------------
# Data & Columns
# -----------------------------
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = ['age','sex','cp','trestbps','chol','fbs','restecg',
           'thalach','exang','oldpeak','slope','ca','thal','target']

NUMERIC = ['age','trestbps','chol','thalach','oldpeak']
CATEGORICAL = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(URL, header=None, names=COLUMNS, na_values='?')
    df = df.dropna()
    df['target'] = (df['target'] > 0).astype(int)
    X = df[NUMERIC + CATEGORICAL]
    y = df['target']
    return X, y

# -----------------------------
# Load saved models
# -----------------------------
@st.cache_resource
def load_saved_models(models_dir="models"):
    paths = sorted(glob.glob(os.path.join(models_dir, "*_pipeline.joblib")))
    models = {}
    for p in paths:
        name = os.path.basename(p).replace("_pipeline.joblib", "")
        models[name] = joblib.load(p)
    return models

# -----------------------------
# Choose best model
# -----------------------------
@st.cache_resource
def choose_best_model(_models):
    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_name = None
    best_score = -1.0
    scores = {}

    for name, model in _models.items():
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores[name] = acc
        if acc > best_score:
            best_name = name
            best_score = acc

    return best_name, best_score, scores

# -----------------------------
# Sidebar: model selection mode
# -----------------------------
models = load_saved_models("models")
if not models:
    st.error("No saved models found in ./models. Expected files like *_pipeline.joblib")
    st.stop()

mode = st.sidebar.radio("Model mode", ["Auto (pick best)", "Manual (choose)"])

if mode == "Auto (pick best)":
    best_name, best_score, all_scores = choose_best_model(models)
    model = models[best_name]
    st.sidebar.success(f"Best model: {best_name}  |  val acc: {best_score:.3f}")
    with st.sidebar.expander("All model scores"):
        for k, v in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            st.write(f"- {k}: {v:.3f}")
else:
    selected = st.sidebar.selectbox("Choose a model", list(models.keys()))
    model = models[selected]
    st.sidebar.info(f"Using model: {selected}")

# -----------------------------
# Sidebar/Form: user input
# -----------------------------
with st.sidebar.form("input_form"):
    st.header("Patient Parameters")

    age = st.slider("Age", 20, 90, 50)
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 220, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 120, 600, 240)
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    oldpeak = st.slider("ST Depression", 0.0, 8.0, 1.0, step=0.1)
    ca = st.slider("Major Vessels colored by Fluoroscopy (ca)", 0, 3, 0)

    sex = st.selectbox("Gender", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox(
        "Chest Pain Type",
        [("Typical Angina", 1), ("Atypical Angina", 2), ("Non-anginal Pain", 3), ("Asymptomatic", 4)],
        format_func=lambda x: x[0],
    )[1]
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [("False", 0), ("True", 1)],
        format_func=lambda x: x[0],
    )[1]
    restecg = st.selectbox(
        "Resting ECG",
        [("Normal", 0), ("ST-T Abnormality", 1), ("LV Hypertrophy", 2)],
        format_func=lambda x: x[0],
    )[1]
    exang = st.selectbox(
        "Exercise-induced Angina",
        [("No", 0), ("Yes", 1)],
        format_func=lambda x: x[0],
    )[1]
    slope = st.selectbox(
        "Slope of Peak ST",
        [("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)],
        format_func=lambda x: x[0],
    )[1]
    thal = st.selectbox(
        "Thalassemia",
        [("Normal", 3), ("Fixed Defect", 6), ("Reversible Defect", 7)],
        format_func=lambda x: x[0],
    )[1]

    submitted = st.form_submit_button("Predict")

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "oldpeak": oldpeak,
        "sex": sex,
        "cp": cp,
        "fbs": fbs,
        "restecg": restecg,
        "exang": exang,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }])

    # Predict probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        prob_disease = float(proba[1])
    else:
        pred = int(model.predict(input_data)[0])
        prob_disease = float(pred)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")
    if prob_disease > 0.5:
        st.error(f"**Disease Likely** — Probability: {prob_disease:.2%}")
    else:
        st.success(f"**Disease Unlikely** — Probability of No Disease: {(1 - prob_disease):.2%}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Tree plot only for DecisionTreeClassifier
    st.markdown('<div class="card" style="margin-top: 16px;">', unsafe_allow_html=True)
    st.subheader("Model Visualization (Decision Tree only)")
    try:
        clf = model.named_steps.get("classifier", None)
        pre = model.named_steps.get("preprocessor", None)

        if isinstance(clf, DecisionTreeClassifier) and pre is not None:
            feature_names = pre.get_feature_names_out()
            fig, ax = plt.subplots(figsize=(25, 12))
            plot_tree(
                clf,
                feature_names=feature_names,
                class_names=["No Disease", "Disease"],
                filled=True,
                rounded=True,
                ax=ax,
                fontsize=10,
            )
            st.pyplot(fig)
        else:
            st.info("Visualization is available only for a single Decision Tree model.")
    except Exception:
        st.info("Visualization not available for this model.")
    st.markdown("</div>", unsafe_allow_html=True)
