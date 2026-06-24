# ensemble_api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import sys
import numpy as np
import shap

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_loader import EnsembleModel
from medical_thresholds import calculate_medical_risk, combine_with_ml

# ==========================================================
# PATHS
# ==========================================================
MODELS_DIR = r"C:\Users\Ruhamah\Desktop\Diabetes risk app\diabtes_app_model_v3\models"

# ==========================================================
# INITIALIZE FASTAPI
# ==========================================================
app = FastAPI(
    title="Diabetes Risk Assessment API",
    description="Weighted Ensemble (10% LR + 90% RF) for Diabetes Risk Prediction",
    version="2.0.0"
)

# Enable CORS for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ensemble model
print("Loading ensemble model...")
model = EnsembleModel(MODELS_DIR)
print("✓ Model loaded successfully")

# ==========================================================
# INITIALIZE SHAP EXPLAINER ONCE AT STARTUP
# ==========================================================
print("Initializing SHAP explainer...")
shap_explainer = shap.TreeExplainer(model.rf_model) 
print("✓ SHAP explainer ready")

# ==========================================================
# REQUEST/RESPONSE MODELS
# ==========================================================
class HealthData(BaseModel):
    gender: int
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int
    activity_minutes: Optional[int] = 0
    family_history: Optional[bool] = False

class RiskResponse(BaseModel):
    ml_risk_score: float
    ml_risk_level: str
    ml_risk_color: str
    ml_recommendation: str
    medical_risk_score: Optional[float] = None
    medical_risk_factors: Optional[list] = None
    final_risk_score: Optional[float] = None
    final_risk_level: Optional[str] = None
    ensemble_details: Dict[str, Any]
    # [FIX 1] Added the shap_values field to the response model so FastAPI accepts it
    shap_values: Optional[Dict[str, float]] = None 

# ==========================================================
# API ENDPOINTS
# ==========================================================
@app.get("/")
def root():
    return {
        "message": "Diabetes Risk Assessment API (Ensemble)",
        "version": "2.0.0",
        "ensemble_weights": f"LR={model.lr_weight}, RF={model.rf_weight}",
        "threshold": model.threshold,
        "features": model.feature_names
    }

@app.get("/health")
def health():
    return {"status": "healthy", "ensemble_loaded": True}

@app.post("/predict", response_model=RiskResponse)
def predict_risk(data: HealthData):
    try:
        # Convert to dictionary
        user_dict = data.dict()
        
        # ML prediction
        ml_result = model.predict(user_dict)
        
        # Medical guidelines (for additional insight)
        medical_result = calculate_medical_risk(user_dict)
        
        # Combined final risk 
        combined = combine_with_ml(
            ml_risk_score=ml_result['risk_score'],
            medical_risk_score=medical_result['medical_risk_score']
        )

        # ==========================================================
        # CALCULATE SHAP VALUES FOR A SPECIFIC USER
        # ==========================================================
        user_shap_dict = {}
        try:
            # 1. Order the user's data exactly as the model expects
            X_user = np.array([[user_dict[feat] for feat in model.feature_names]])
            
            # Scale the data before passing to SHAP!
            X_user_scaled = model.scaler.transform(X_user)
            
            # 2. Calculate SHAP values using the SCALED data
            shap_vals = shap_explainer.shap_values(X_user_scaled)
            
            # 3. Extract the values for the positive class (Diabetes)
            # TreeExplainer returns a list for binary classification [class_0, class_1]
            if isinstance(shap_vals, list):
                user_shap_vals = shap_vals[1][0]
            else:
                # Fallback for different SHAP versions that return a 3D array
                user_shap_vals = shap_vals[0, :, 1] if len(shap_vals.shape) == 3 else shap_vals[0]
                
            # 4. Map the numpy array back to a clean dictionary of feature names
            for i, feat_name in enumerate(model.feature_names):
                user_shap_dict[feat_name] = float(user_shap_vals[i])
                
        # except block
        except Exception as shap_e:
            # Fail gracefully: If SHAP fails for some reason, don't crash the whole prediction
            print(f"Warning: SHAP calculation failed: {shap_e}")
            user_shap_dict = None

        return RiskResponse(
            ml_risk_score=ml_result['risk_score'],
            ml_risk_level=ml_result['risk_level'],
            ml_risk_color=ml_result['risk_color'],
            ml_recommendation=ml_result['recommendation'],
            medical_risk_score=medical_result['medical_risk_score'],
            medical_risk_factors=medical_result['risk_factors'],
            final_risk_score=combined['final_risk_score'],
            final_risk_level=combined['final_risk_level'],
            ensemble_details=ml_result['ensemble_details'],
            shap_values=user_shap_dict 
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/features")
def get_features():
    return {
        "features": model.feature_names,
        "count": len(model.feature_names),
        "ensemble_weights": {
            "logistic_regression": model.lr_weight, 
            "random_forest": model.rf_weight
        },
        "threshold": model.threshold
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 Starting Ensemble API")
    print("="*50)
    print(f"📡 API running at: http://localhost:8001")
    print(f"📚 API docs at: http://localhost:8001/docs")
    print(f"🔍 Health check: http://localhost:8001/health")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=8001)