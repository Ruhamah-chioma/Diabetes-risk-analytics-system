# train_kaggle_ensemble.py
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import shap

# ==========================================================
# PATHS
# ==========================================================
processed_dir = r"C:\Users\Ruhamah\Desktop\Diabetes risk app\diabtes_app_model_v3\dataset\processed"
models_dir = r"C:\Users\Ruhamah\Desktop\Diabetes risk app\diabtes_app_model_v3\models"
shap_dir = r"C:\Users\Ruhamah\Desktop\Diabetes risk app\diabtes_app_model_v3\shap_plots"

print("=" * 60)
print("WEIGHTED ENSEMBLE (10% LR + 90% RF) + SHAP")
print("=" * 60)

# ==========================================================
# LOAD DATA
# ==========================================================
print("\n1. Loading preprocessed data...")

X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
X_val   = np.load(os.path.join(processed_dir, "X_val.npy"))
X_test  = np.load(os.path.join(processed_dir, "X_test.npy"))

y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
y_val   = np.load(os.path.join(processed_dir, "y_val.npy"))
y_test  = np.load(os.path.join(processed_dir, "y_test.npy"))

feature_names = joblib.load(os.path.join(processed_dir, "feature_names.pkl"))

print(f"   Training:   {X_train.shape}")
print(f"   Validation: {X_val.shape}")
print(f"   Test:       {X_test.shape}")
print(f"   Features:   {feature_names}")

# ==========================================================
# TRAIN LOGISTIC REGRESSION
# ==========================================================
print("\n2. Training Logistic Regression...")

lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight="balanced"
)

lr_model.fit(X_train, y_train)
print("   ✓ Logistic Regression training complete")

# ==========================================================
# TRAIN RANDOM FOREST
# ==========================================================
print("\n3. Training Random Forest Classifier...")

rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("   ✓ Random Forest training complete")

# ==========================================================
# VALIDATION EVALUATION (INDIVIDUAL MODELS)
# ==========================================================
print("\n4. Validation Evaluation (Individual Models)...")

# Logistic Regression
lr_val_proba = lr_model.predict_proba(X_val)[:, 1]
lr_val_acc = accuracy_score(y_val, (lr_val_proba >= 0.5).astype(int))
lr_val_auc = roc_auc_score(y_val, lr_val_proba)
print(f"   Logistic Regression - Acc: {lr_val_acc*100:.2f}%, AUC: {lr_val_auc:.4f}")

# Random Forest
rf_val_proba = rf_model.predict_proba(X_val)[:, 1]
rf_val_acc = accuracy_score(y_val, (rf_val_proba >= 0.5).astype(int))
rf_val_auc = roc_auc_score(y_val, rf_val_proba)
print(f"   Random Forest       - Acc: {rf_val_acc*100:.2f}%, AUC: {rf_val_auc:.4f}")

# ==========================================================
# WEIGHTED ENSEMBLE (10% LR + 90% RF)
# ==========================================================
print("\n5. Weighted Ensemble (10% LR + 90% RF)...")

# Ensemble weights
lr_weight = 0.1
rf_weight = 0.9

# Ensemble predictions on validation set
ensemble_val_proba = (lr_weight * lr_val_proba) + (rf_weight * rf_val_proba)

# Find best threshold using F1-score
print("\n   Optimising decision threshold using F1-score...")
print("-" * 70)
print(" Threshold | Accuracy | Precision | Recall | F1")
print("-" * 70)

best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.30, 0.71, 0.02):
    val_pred = (ensemble_val_proba >= threshold).astype(int)

    acc = accuracy_score(y_val, val_pred)
    prec = precision_score(y_val, val_pred, zero_division=0)
    rec = recall_score(y_val, val_pred, zero_division=0)
    f1 = f1_score(y_val, val_pred, zero_division=0)

    print(f"   {threshold:.2f}    | {acc:.4f}   | {prec:.4f}    | {rec:.4f} | {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("-" * 70)
print(f"   Best Threshold = {best_threshold:.2f}")
print(f"   Best Validation F1 = {best_f1:.4f}")

# ==========================================================
# FINAL TEST EVALUATION (ENSEMBLE)
# ==========================================================
print("\n6. Final Test Evaluation (Ensemble)...")

# Get predictions from both models on test set
lr_test_proba = lr_model.predict_proba(X_test)[:, 1]
rf_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Weighted ensemble
ensemble_test_proba = (lr_weight * lr_test_proba) + (rf_weight * rf_test_proba)
test_pred = (ensemble_test_proba >= best_threshold).astype(int)

# Metrics
test_acc = accuracy_score(y_test, test_pred)
test_auc = roc_auc_score(y_test, ensemble_test_proba)
test_precision = precision_score(y_test, test_pred, zero_division=0)
test_recall = recall_score(y_test, test_pred, zero_division=0)
test_f1 = f1_score(y_test, test_pred, zero_division=0)

print(f"\n   Ensemble Test Results:")
print(f"   Test Accuracy : {test_acc*100:.2f}%")
print(f"   Test AUC      : {test_auc:.4f}")
print(f"   Test Precision: {test_precision:.4f}")
print(f"   Test Recall   : {test_recall:.4f}")
print(f"   Test F1 Score : {test_f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, test_pred)
print("\n   Confusion Matrix:")
print(f"   TP: {cm[1,1]} | FP: {cm[0,1]}")
print(f"   FN: {cm[1,0]} | TN: {cm[0,0]}")

# Classification Report
print("\n7. Classification Report (Ensemble):")
print(classification_report(y_test, test_pred, target_names=["No Diabetes", "Diabetes"]))

# ==========================================================
# FEATURE IMPORTANCE (FROM RANDOM FOREST)
# ==========================================================
print("\n8. Feature Importance (Random Forest Component):")

importance = list(zip(feature_names, rf_model.feature_importances_))
importance.sort(key=lambda x: x[1], reverse=True)

for i, (feature, score) in enumerate(importance, start=1):
    print(f"   {i}. {feature}: {score*100:.2f}%")

# ==========================================================
# SHAP EXPLAINABILITY (FOR RANDOM FOREST COMPONENT)
# ==========================================================
print("\n9. Generating SHAP explanations (for Random Forest component)...")
os.makedirs(shap_dir, exist_ok=True)

# Initialize SHAP explainer for Random Forest
explainer = shap.TreeExplainer(rf_model)
print("   ✓ SHAP explainer created")

# Compute SHAP values for a subset
n_samples = min(300, len(X_test))
X_test_sample = X_test[:n_samples]
print(f"   Computing SHAP values for {n_samples} samples...")
shap_values = explainer.shap_values(X_test_sample)
print("   ✓ SHAP values computed")

# Get class 1 (Diabetes) SHAP values
if isinstance(shap_values, list):
    shap_values_diabetes = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values_diabetes = shap_values[:, :, 1]
else:
    shap_values_diabetes = shap_values

# SHAP Summary Plot
print("\n   Generating SHAP summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values_diabetes, 
    X_test_sample, 
    feature_names=feature_names, 
    show=False, 
    plot_type="dot"
)
plt.title("SHAP Feature Importance - Diabetes Risk Prediction", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(shap_dir, "shap_summary_plot.png"), dpi=300, bbox_inches="tight")
plt.close()
print("      ✓ Saved: shap_summary_plot.png")

# SHAP Bar Plot
print("\n   Generating SHAP bar plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_diabetes, 
    X_test_sample, 
    feature_names=feature_names, 
    show=False, 
    plot_type="bar"
)
plt.title("Mean SHAP Values per Feature", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(shap_dir, "shap_bar_plot.png"), dpi=300, bbox_inches="tight")
plt.close()
print("      ✓ Saved: shap_bar_plot.png")

# SHAP Feature Importance Ranking
print("\n   SHAP Feature Importance Ranking (Mean |SHAP|):")
shap_importance = np.abs(shap_values_diabetes).mean(axis=0)

shap_ranking = []
for i, feature in enumerate(feature_names):
    score = float(shap_importance[i])
    shap_ranking.append((feature, score))

shap_ranking.sort(key=lambda x: x[1], reverse=True)

for i, (feature, score) in enumerate(shap_ranking, start=1):
    print(f"      {i}. {feature}: {score:.6f}")

# ==========================================================
# SAVE MODELS AND CONFIGURATION
# ==========================================================
print("\n10. Saving models and configuration...")
os.makedirs(models_dir, exist_ok=True)

# Save individual models
joblib.dump(lr_model, os.path.join(models_dir, "logistic_regression_model.pkl"))
joblib.dump(rf_model, os.path.join(models_dir, "random_forest_model.pkl"))

# Save ensemble configuration
config = {
    "model_type": "WeightedEnsemble",
    "lr_weight": lr_weight,
    "rf_weight": rf_weight,
    "threshold": best_threshold,
    "feature_names": feature_names,
    "test_accuracy": float(test_acc),
    "test_auc": float(test_auc),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "test_f1": float(test_f1)
}
joblib.dump(config, os.path.join(models_dir, "ensemble_config.pkl"))

print(f"   ✓ Models and config saved to {models_dir}")

# ==========================================================
# FINAL SUMMARY
# ==========================================================
print("\n" + "=" * 60)
print("✅ WEIGHTED ENSEMBLE + SHAP COMPLETE!")
print("=" * 60)

print(f"\n📊 FINAL ENSEMBLE RESULTS")
print(f"   Ensemble Weights: LR={lr_weight}, RF={rf_weight}")
print(f"   Best Threshold   : {best_threshold:.2f}")
print(f"   Accuracy         : {test_acc*100:.2f}%")
print(f"   AUC              : {test_auc:.4f}")
print(f"   Precision        : {test_precision:.4f}")
print(f"   Recall           : {test_recall:.4f}")
print(f"   F1 Score         : {test_f1:.4f}")

print(f"\n📁 SHAP PLOTS SAVED TO: {shap_dir}")
print("   - shap_summary_plot.png")
print("   - shap_bar_plot.png")
print("=" * 60)