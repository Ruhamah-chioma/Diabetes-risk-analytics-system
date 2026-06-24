# model_loader.py
import joblib
import numpy as np
import os

class EnsembleModel:
    def __init__(self, models_dir):
        """Load the weighted ensemble model and the scaler"""
        self.models_dir = models_dir
        
        # 1. Locate the processed directory to load the scaler
        base_dir = os.path.dirname(models_dir)
        processed_dir = os.path.join(base_dir, "dataset", "processed")
        
        # Load individual models
        self.lr_model = joblib.load(os.path.join(models_dir, "logistic_regression_model.pkl"))
        self.rf_model = joblib.load(os.path.join(models_dir, "random_forest_model.pkl"))
        
        # 2. Load the scaler!
        scaler_path = os.path.join(processed_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("✓ Scaler loaded successfully")
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please check your paths.")
        
        # Load ensemble configuration
        config = joblib.load(os.path.join(models_dir, "ensemble_config.pkl"))
        self.lr_weight = config["lr_weight"]
        self.rf_weight = config["rf_weight"]
        self.threshold = config["threshold"]
        self.feature_names = config["feature_names"]
        
        print(f"✓ Ensemble loaded: LR={self.lr_weight}, RF={self.rf_weight}")
        print(f"✓ Threshold: {self.threshold}")
        print(f"✓ Features: {self.feature_names}")
    
    def predict(self, user_data):
        """
        user_data: dictionary with 8 health features
        Returns: risk score and detailed prediction
        """
        # Create feature array in correct order
        features = []
        for feature in self.feature_names:
            features.append(user_data[feature])
        
        features_array = np.array(features).reshape(1, -1)
        
        # 3. SCALE THE DATA BEFORE PREDICTING
        # This transforms raw values (e.g., glucose=85) into the scaled space (e.g., -0.75)
        features_array_scaled = self.scaler.transform(features_array)
        
        # Get individual predictions using the SCALED data
        lr_proba = self.lr_model.predict_proba(features_array_scaled)[0][1]
        rf_proba = self.rf_model.predict_proba(features_array_scaled)[0][1]
        
        # Weighted ensemble
        ensemble_proba = (self.lr_weight * lr_proba) + (self.rf_weight * rf_proba)
        risk_score = ensemble_proba * 100
        
        # Determine risk level
        if risk_score >= 65:
            risk_level = "High Risk"
            color = "#e74c3c"
            recommendation = "Please consult a healthcare provider for a comprehensive diabetes screening."
        elif risk_score >= 40:
            risk_level = "Medium Risk (Prediabetes Range)"
            color = "#f39c12"
            recommendation = "Consider lifestyle modifications and monitor your blood glucose regularly."
        else:
            risk_level = "Low Risk"
            color = "#2ecc71"
            recommendation = "Maintain your healthy lifestyle with regular exercise and balanced diet."
        
        return {
            "risk_score": round(risk_score, 1),
            "risk_percentage": f"{round(risk_score, 1)}%",
            "risk_level": risk_level,
            "risk_color": color,
            "recommendation": recommendation,
            "ensemble_details": {
                "logistic_regression": round(lr_proba * 100, 1),
                "random_forest": round(rf_proba * 100, 1),
                "lr_weight": self.lr_weight,
                "rf_weight": self.rf_weight
            }
        }
    
    def get_feature_names(self):
        return self.feature_names

# For testing
if __name__ == "__main__":
    models_dir = r"C:\Users\Ruhamah\Desktop\Diabetes risk app\diabtes_app_model_v3\models"
    model = EnsembleModel(models_dir)
    
    # Test with sample data (Your exact curl input)
    sample_user = {
        'gender': 0,
        'age': 19,
        'hypertension': 0,
        'heart_disease': 0,
        'smoking_history': 0,
        'bmi': 23.4,
        'HbA1c_level': 5.2,
        'blood_glucose_level': 85
    }
    
    result = model.predict(sample_user)
    print("="*50)
    print("SAMPLE PREDICTION")
    print("="*50)
    print(f"Risk Score: {result['risk_score']}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"LR: {result['ensemble_details']['logistic_regression']}%")
    print(f"RF: {result['ensemble_details']['random_forest']}%")