# Diabetes Risk Analytics and Visualisation System

A mobile health application that predicts diabetes risk using a weighted ensemble of Logistic Regression and Random Forest, with SHAP explainability for model interpretability.

## 📱 Features

- User authentication (signup/login with security questions)
- Health data input with real-time validation
- Diabetes risk prediction using weighted ensemble (10% LR + 90% RF)
- SHAP explainability for individual risk scores
- Risk trend visualisation and historical tracking
- 4-week reminder notifications
- Dark/light theme support

## 🏗️ Tech Stack

- **Frontend:** React Native (Expo)
- **Backend:** FastAPI (Python)
- **Machine Learning:** scikit-learn (Logistic Regression, Random Forest)
- **Explainability:** SHAP
- **Storage:** AsyncStorage (local)

## 📊 Model Performance

- **Test Accuracy:** 95.71%
- **AUC:** 0.9724
- **Recall:** 77.82%
- **Precision:** 75.08%

## 🔧 Installation

### Backend
```bash
cd backend
pip install -r requirements.txt
python ensemble_api.py