# utils/medical_thresholds.py
"""
Medical thresholds based on American Diabetes Association (ADA) standards
These are STANDARD MEDICAL GUIDELINES, not ML predictions
"""

# ==============================================
# STANDARD MEDICAL THRESHOLDS (ADA 2024)
# ==============================================

# Blood Glucose thresholds (mg/dL)
# Source: American Diabetes Association
BLOOD_GLUCOSE = {
    'normal': {'max': 99, 'label': 'Normal', 'risk_points': 0},
    'prediabetes': {'min': 100, 'max': 125, 'label': 'Prediabetes', 'risk_points': 25},
    'diabetes': {'min': 126, 'label': 'Diabetes', 'risk_points': 50}
}

# BMI thresholds (kg/m²)
# Source: World Health Organization / CDC
BMI = {
    'underweight': {'max': 18.4, 'label': 'Underweight', 'risk_points': 5},
    'normal': {'min': 18.5, 'max': 24.9, 'label': 'Normal', 'risk_points': 0},
    'overweight': {'min': 25, 'max': 29.9, 'label': 'Overweight', 'risk_points': 15},
    'obese_class1': {'min': 30, 'max': 34.9, 'label': 'Obese Class I', 'risk_points': 25},
    'obese_class2': {'min': 35, 'max': 39.9, 'label': 'Obese Class II', 'risk_points': 35},
    'obese_class3': {'min': 40, 'label': 'Obese Class III', 'risk_points': 45}
}

# Age risk thresholds
# Source: ADA - Risk increases with age
AGE_RISK = {
    'young': {'max': 44, 'risk_points': 0},
    'middle': {'min': 45, 'max': 64, 'risk_points': 15},
    'senior': {'min': 65, 'risk_points': 25}
}

# Physical Activity (minutes per week)
# Source: CDC recommendations
ACTIVITY = {
    'sedentary': {'minutes': 0, 'risk_points': 20, 'label': 'No regular exercise'},
    'insufficient': {'minutes': 149, 'risk_points': 10, 'label': 'Some exercise but insufficient'},
    'active': {'minutes': 150, 'risk_points': 0, 'label': 'Regular exercise (good)'},
    'very_active': {'minutes': 300, 'risk_points': -5, 'label': 'Very active (excellent)'}
}

# Blood Pressure thresholds (mmHg)
# Source: American Heart Association
BLOOD_PRESSURE = {
    'normal': {'systolic_max': 120, 'diastolic_max': 80, 'risk_points': 0},
    'elevated': {'systolic_min': 120, 'systolic_max': 129, 'diastolic_max': 80, 'risk_points': 10},
    'hypertension_stage1': {'systolic_min': 130, 'systolic_max': 139, 'diastolic_min': 80, 'diastolic_max': 89, 'risk_points': 20},
    'hypertension_stage2': {'systolic_min': 140, 'diastolic_min': 90, 'risk_points': 30}
}

# Family History (first-degree relative with diabetes)
# Source: ADA
FAMILY_HISTORY_RISK = 15  # Adds 15 points if yes

# ==============================================
# MAIN FUNCTION
# ==============================================

def calculate_medical_risk(user_data):
    """
    Calculate risk score based on standard medical guidelines ONLY.
    
    user_data should contain:
    - glucose: blood glucose level in mg/dL
    - bmi: body mass index
    - age: age in years
    - activity_minutes: minutes of exercise per week
    - family_history: True/False (optional)
    - systolic_bp: systolic blood pressure (optional)
    - diastolic_bp: diastolic blood pressure (optional)
    
    Returns risk score (0-100) and list of risk factors.
    """
    risk_points = 0
    risk_factors = []
    
    # 1. Blood Glucose (most important)
    glucose = user_data.get('glucose', 0)
    for level, thresholds in BLOOD_GLUCOSE.items():
        if 'max' in thresholds and glucose <= thresholds['max']:
            risk_points += thresholds['risk_points']
            if thresholds['risk_points'] > 0:
                risk_factors.append(f"🩸 Blood glucose: {glucose} mg/dL ({thresholds['label']})")
            break
        elif 'min' in thresholds and glucose >= thresholds['min']:
            if 'max' not in thresholds or glucose <= thresholds['max']:
                risk_points += thresholds['risk_points']
                risk_factors.append(f"🩸 Blood glucose: {glucose} mg/dL ({thresholds['label']})")
                break
    
    # 2. BMI
    bmi = user_data.get('bmi', 0)
    for level, thresholds in BMI.items():
        if 'max' in thresholds and bmi <= thresholds['max']:
            risk_points += thresholds['risk_points']
            if thresholds['risk_points'] > 0:
                risk_factors.append(f"⚖️ BMI: {bmi} ({thresholds['label']})")
            break
        elif 'min' in thresholds and bmi >= thresholds['min']:
            if 'max' not in thresholds or bmi <= thresholds['max']:
                risk_points += thresholds['risk_points']
                if thresholds['risk_points'] > 0:
                    risk_factors.append(f"⚖️ BMI: {bmi} ({thresholds['label']})")
                break
    
    # 3. Age
    age = user_data.get('age', 0)
    for level, thresholds in AGE_RISK.items():
        if 'max' in thresholds and age <= thresholds['max']:
            risk_points += thresholds['risk_points']
            break
        elif 'min' in thresholds and age >= thresholds['min']:
            if 'max' not in thresholds or age <= thresholds['max']:
                risk_points += thresholds['risk_points']
                if thresholds['risk_points'] > 0:
                    risk_factors.append(f"📅 Age: {age} years (risk increases after 45)")
                break
    
    # 4. Physical Activity
    activity_minutes = user_data.get('activity_minutes', 0)
    for level, thresholds in ACTIVITY.items():
        if activity_minutes <= thresholds['minutes']:
            risk_points += thresholds['risk_points']
            if thresholds['risk_points'] > 0:
                risk_factors.append(f"🏃 Activity: {thresholds['label']}")
            elif thresholds['risk_points'] < 0:
                risk_factors.append(f"🏃 Activity: {thresholds['label']} ✨ reduces risk")
            break
    
    # 5. Family History (if provided)
    if user_data.get('family_history', False):
        risk_points += FAMILY_HISTORY_RISK
        risk_factors.append(f"👨‍👩‍👧 Family history of diabetes (+{FAMILY_HISTORY_RISK} points)")
    
    # 6. Blood Pressure (if provided)
    if 'systolic_bp' in user_data and 'diastolic_bp' in user_data:
        systolic = user_data['systolic_bp']
        diastolic = user_data['diastolic_bp']
        for level, thresholds in BLOOD_PRESSURE.items():
            match = True
            if 'systolic_max' in thresholds and systolic > thresholds['systolic_max']:
                match = False
            if 'systolic_min' in thresholds and systolic < thresholds['systolic_min']:
                match = False
            if 'diastolic_max' in thresholds and diastolic > thresholds['diastolic_max']:
                match = False
            if 'diastolic_min' in thresholds and diastolic < thresholds['diastolic_min']:
                match = False
            
            if match:
                risk_points += thresholds['risk_points']
                if thresholds['risk_points'] > 0:
                    risk_factors.append(f"❤️ Blood pressure: {systolic}/{diastolic} ({level})")
                break
    
    # Cap risk points at 0-100
    risk_points = max(0, min(100, risk_points))
    
    # Determine risk level
    if risk_points >= 60:
        risk_level = "High Risk"
    elif risk_points >= 40:
        risk_level = "Medium Risk (Prediabetes Range)"
    else:
        risk_level = "Low Risk"
    
    return {
        'medical_risk_score': risk_points,
        'medical_risk_level': risk_level,
        'risk_factors': risk_factors,
        'recommendations': _get_recommendations(risk_factors)
    }


def _get_recommendations(risk_factors):
    """Generate personalized recommendations based on risk factors"""
    recommendations = []
    
    for factor in risk_factors:
        if 'glucose' in factor.lower():
            if 'prediabetes' in factor.lower():
                recommendations.append("📋 Monitor blood glucose regularly and consult your doctor")
            elif 'diabetes' in factor.lower():
                recommendations.append("⚠️ Please consult a healthcare provider immediately")
        if 'bmi' in factor.lower() and ('overweight' in factor.lower() or 'obese' in factor.lower()):
            recommendations.append("🥗 Consider dietary changes and increased physical activity")
        if 'age' in factor.lower():
            recommendations.append("📅 Regular diabetes screening is recommended")
        if 'activity' in factor.lower() and ('sedentary' in factor.lower() or 'insufficient' in factor.lower()):
            recommendations.append("🚶 Start with 30 minutes of walking, 5 days per week")
        if 'blood pressure' in factor.lower():
            recommendations.append("❤️ Monitor blood pressure and reduce sodium intake")
        if 'family history' in factor.lower():
            recommendations.append("👨‍👩‍👧 Regular screening is especially important due to family history")
    
    if not recommendations:
        recommendations.append("✅ Maintain healthy lifestyle with regular exercise and balanced diet")
    
    return recommendations


def combine_with_ml(ml_risk_score, medical_risk_score, ml_weight=0.7, medical_weight=0.3):
    """Combine ML and medical risk scores"""
    final_risk = (ml_risk_score * ml_weight) + (medical_risk_score * medical_weight)
    
    if final_risk >= 60:
        final_level = "High Risk"
    elif final_risk >= 40:
        final_level = "Medium Risk (Prediabetes Range)"
    else:
        final_level = "Low Risk"
    
    return {
        'final_risk_score': round(final_risk, 1),
        'final_risk_level': final_level,
        'ml_contribution': round(ml_risk_score, 1),
        'medical_contribution': round(medical_risk_score, 1)
    }

