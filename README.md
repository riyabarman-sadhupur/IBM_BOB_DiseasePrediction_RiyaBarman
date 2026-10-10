# 🏥 Disease Prediction System

**Author:** Riya Barman  
**Submission Files:** `RiyaBarman_DiseasePrediction.ipynb` · `requirements.txt` · `RiyaBarman_DiseasePredictionReport.docx` · `README.md`

---

## 📌 Project Description

An end-to-end machine learning application that predicts the likelihood of **cardiovascular disease** in patients based on 10 clinical features. The system is built with a clean 3-layer architecture:

| Layer | Technology | Role |
|-------|-----------|------|
| ML Pipeline | NumPy, Pandas, Matplotlib, Seaborn | Dataset generation, model training, EDA |
| Backend API | Flask 3.x | REST API serving live predictions |
| Frontend UI | Streamlit | Interactive 5-page web dashboard |

The project implements **Logistic Regression** and **Random Forest** entirely from scratch using pure NumPy (no scikit-learn), making it compatible with Python 3.14 and any environment where compiled ML libraries are unavailable.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Synthetically generated (clinically-realistic) |
| Reference | Inspired by [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Kaggle alternative | [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) |
| Rows | 1,200 patients |
| Features | 10 clinical features |
| Target | `0` = No Disease, `1` = Disease |
| Prevalence | ~24% disease cases |

### Features Used

| # | Feature | Type | Range |
|---|---------|------|-------|
| 1 | `age` | Numeric | 25–80 years |
| 2 | `sex` | Binary | 0 = Female, 1 = Male |
| 3 | `bmi` | Numeric | 15–50 |
| 4 | `blood_pressure` | Numeric | 80–200 mmHg |
| 5 | `cholesterol` | Numeric | 120–350 mg/dL |
| 6 | `glucose` | Numeric | 60–300 mg/dL |
| 7 | `smoking` | Binary | 0 = No, 1 = Yes |
| 8 | `family_history` | Binary | 0 = No, 1 = Yes |
| 9 | `chest_pain_type` | Categorical | 0–3 |
| 10 | `heart_rate` | Numeric | 45–130 bpm |

---

## 🧠 Models & Results

| Model | Accuracy | F1 Score | ROC-AUC | CV Acc |
|-------|----------|----------|---------|--------|
| **Logistic Regression** ⭐ | **79.17%** | **0.5455** | **0.8434** | **0.849 ± 0.022** |
| Random Forest | 76.25% | 0.0656 | 0.8025 | — |

**Best model: Logistic Regression** (selected by ROC-AUC)

---

## 🗂️ Project Structure

```
disease_prediction/
├── RiyaBarman_DiseasePrediction.ipynb     ← Complete Jupyter Notebook (all code)
├── requirements.txt                        ← Python dependencies
├── RiyaBarman_DiseasePredictionReport.docx ← Full project report (Word)
├── README.md                               ← This file
│
├── train_model.py          — Standalone training script
├── models_def.py           — Shared model class definitions (pickle-safe)
├── app.py                  — Flask REST API backend
├── ui.py                   — Streamlit frontend UI (5 pages)
├── generate_screenshots.py — UI mockup generator (matplotlib)
├── generate_report.py      — Word report builder
│
├── data/
│   └── dataset.csv         — Generated dataset (1,200 rows × 11 cols)
├── models/
│   ├── best_model.pkl       — Trained Logistic Regression model
│   ├── scaler.pkl           — Fitted StandardScaler
│   └── model_metadata.json  — Metrics and feature info
└── report_assets/           — Auto-generated plots & UI screenshots
    ├── eda_distribution.png
    ├── eda_boxplots.png
    ├── correlation_heatmap.png
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── lr_loss_curve.png
    ├── ui_home.png
    ├── ui_predict.png
    ├── ui_model_insights.png
    └── ui_dataset.png
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8 or higher (tested on Python 3.14)
- pip

### 1. Install dependencies
```bash
cd disease_prediction
pip install -r requirements.txt
```

---

## 🚀 Run Instructions

### Step 1 — Run the Jupyter Notebook (all-in-one)
```bash
jupyter notebook RiyaBarman_DiseasePrediction.ipynb
```
Run all cells top-to-bottom. This generates the dataset, trains models, saves artifacts, and shows all plots.

### Step 2 — (Optional) Start Flask API in a terminal
```bash
python app.py
```
API will be live at → **`http://127.0.0.1:5000`**

### Step 3 — (Optional) Launch Streamlit UI in another terminal
```bash
streamlit run ui.py
```
Dashboard will open at → **`http://localhost:8501`**

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/api/model-info` | Model metrics & metadata |
| `GET` | `/api/dataset-stats` | Dataset statistics |
| `GET` | `/api/features` | Feature list & descriptions |
| `POST` | `/api/predict` | Single patient prediction |
| `POST` | `/api/predict-batch` | Bulk prediction (JSON array) |

### Quick Test
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"bmi":30.0,"blood_pressure":145,"cholesterol":240,
       "glucose":115,"smoking":1,"family_history":1,"chest_pain_type":2,"heart_rate":88}'
```
Response: `{"label":"Disease","prediction":1,"probability":0.8958,"risk_level":"High"}`

---

## 🖥️ Streamlit UI Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | KPI cards, class distribution pie, system overview |
| 🔮 **Predict Disease** | Patient form → real-time prediction with risk badge |
| 📊 **Model Insights** | Performance charts, model comparison table |
| 📁 **Dataset Explorer** | Feature statistics, distribution charts |
| 📦 **Batch Predict** | Upload CSV or paste JSON for bulk predictions |

---

## 🛠️ Technologies Used

- **Python 3.14** — runtime
- **NumPy** — pure-NumPy ML implementations (Logistic Regression, Random Forest, metrics)
- **Pandas** — data manipulation
- **Matplotlib / Seaborn** — visualisation & EDA plots
- **Flask 3.x** — REST API backend
- **Streamlit** — interactive web dashboard frontend
- **python-docx** — Word report generation
- **pickle** (stdlib) — model serialisation

---

## 📝 Key Design Decisions

1. **No scikit-learn** — all ML algorithms (LR, RF, metrics) are implemented from scratch in NumPy for maximum portability and Python 3.14 compatibility.
2. **`models_def.py` shared module** — custom classes are defined in one place and imported by both `train_model.py` and `app.py`, ensuring `pickle` can correctly deserialise saved models.
3. **Flask + Streamlit separation** — the inference layer (Flask) is independent of the UI (Streamlit), allowing both to be tested, scaled, or replaced independently.

---

*Built with Python · Flask · Streamlit · IBM Bob*
