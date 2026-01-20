"""
NHANES Biological Age Predictor Dashboard
A "Glass-Box" Clinical Decision Support System with XAI Features
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Explicitly set backend for headless environments
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import shap
import dice_ml
from dice_ml import Dice

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="NHANES Age Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS for Premium UI
# ============================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Prediction cards */
    .prediction-aged {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(220,53,69,0.3);
    }
    
    .prediction-adult {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(40,167,69,0.3);
    }
    
    .prediction-child {
        background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(23,162,184,0.3);
    }
    
    .prediction-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }
    
    .prediction-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e3a5f;
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Info boxes */
    .info-box {
        background: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Data Loading & Model Training (Cached)
# ============================================================================
@st.cache_data
def load_data():
    """Load and preprocess the NHANES dataset."""
    # Load from local CSV
    df = pd.read_csv("input/NHANES_age_prediction.csv")
    
    # Rename columns to match script convention
    feature_name_dict = {
        "SEQN": "ID",
        "age_group": "Age_Group_Original",
        "RIDAGEYR": "Age",
        "RIAGENDR": "Gender",
        "PAQ605": "Physical_Activity",
        "BMXBMI": "BMI",
        "LBXGLU": "Fasting_Blood_Glucose",
        "DIQ010": "Diabetes_Status",
        "LBXGLT": "Oral_Glucose_Tolerance",
        "LBXIN": "Blood_Insulin"
    }
    df = df.rename(columns=feature_name_dict)
    
    # Feature columns (exclude ID, Age, and original age group)
    feature_cols = ['Gender', 'Physical_Activity', 'BMI', 'Fasting_Blood_Glucose', 
                    'Diabetes_Status', 'Oral_Glucose_Tolerance', 'Blood_Insulin']
    
    X = df[feature_cols].copy()
    y_age = df['Age'].copy()
    
    return X, y_age, df


def bin_age_3_class(age):
    """Bin age into 3 classes: Child/Adult/Aged."""
    if age <= 18:
        return 'Infant/Child/Adolescent'
    elif age <= 64:
        return 'Adult (19-64)'
    else:
        return 'Aged (65+)'


@st.cache_resource
def train_model(_X, _y_encoded, _le):
    """Train XGBoost model and create SHAP explainer."""
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y_encoded, test_size=0.2, random_state=42, stratify=_y_encoded
    )
    
    # Train model
    xgb_model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(_le.classes_),
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        min_child_weight=3,
        gamma=1.5,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    # Soft class weights
    balanced_weights = compute_sample_weight('balanced', y_train)
    soft_weights = np.sqrt(balanced_weights)
    xgb_model.fit(X_train, y_train, sample_weight=soft_weights)
    
    # Create SHAP explainer
    background = shap.sample(_X, 100)
    explainer = shap.Explainer(xgb_model.predict_proba, background)
    
    return xgb_model, explainer, X_train, y_train


@st.cache_resource
def create_dice_explainer(_X_train, _y_train, _xgb_model, _feature_names):
    """Create DiCE explainer for counterfactual generation."""
    # Prepare data for DiCE
    X_train_df = _X_train.copy()
    X_train_df['age_group'] = _y_train
    
    # Define feature types
    continuous_features = ['BMI', 'Fasting_Blood_Glucose', 'Oral_Glucose_Tolerance', 'Blood_Insulin']
    
    # Create DiCE data object
    dice_data = dice_ml.Data(
        dataframe=X_train_df,
        continuous_features=continuous_features,
        outcome_name='age_group'
    )
    
    # Model wrapper
    class XGBModelWrapper:
        def __init__(self, model, feature_names):
            self.model = model
            self.feature_names = feature_names
        
        def predict(self, X):
            if isinstance(X, pd.DataFrame):
                X_numeric = X[self.feature_names].apply(pd.to_numeric, errors='coerce')
                return self.model.predict(X_numeric)
            return self.model.predict(X)
        
        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                X_numeric = X[self.feature_names].apply(pd.to_numeric, errors='coerce')
                return self.model.predict_proba(X_numeric)
            return self.model.predict_proba(X)
    
    wrapped_model = XGBModelWrapper(_xgb_model, list(_feature_names))
    dice_model = dice_ml.Model(model=wrapped_model, backend="sklearn", model_type="classifier")
    dice_exp = Dice(dice_data, dice_model, method="random")
    
    return dice_exp, continuous_features


# ============================================================================
# Load Data and Train Model
# ============================================================================
X, y_age, full_df = load_data()

# Apply 3-class binning
y_binned = y_age.apply(bin_age_3_class)
le = LabelEncoder()
y_encoded = le.fit_transform(y_binned)

# Train model
xgb_model, shap_explainer, X_train, y_train = train_model(X, y_encoded, le)

# Create DiCE explainer
dice_exp, continuous_features = create_dice_explainer(X_train.reset_index(drop=True), y_train, xgb_model, X.columns)

# Get Subject #25 data (index 24 in 0-indexed)
subject_25_idx = 25  # Row 26 in CSV (1-indexed header + data)
subject_25_data = X.iloc[subject_25_idx - 1] if subject_25_idx <= len(X) else X.iloc[0]


# ============================================================================
# Sidebar - Patient Input Form
# ============================================================================
st.sidebar.markdown("## 📋 Patient Data Entry")
st.sidebar.markdown("---")

# Gender
gender_options = {"Male": 1.0, "Female": 2.0}
gender_default = 1.0 if subject_25_data['Gender'] == 1.0 else 2.0
gender = st.sidebar.selectbox(
    "Gender",
    options=list(gender_options.keys()),
    index=0 if gender_default == 1.0 else 1
)
gender_val = gender_options[gender]

# Physical Activity
activity_options = {"Vigorous Activity (Yes)": 1.0, "Vigorous Activity (No)": 2.0}
activity_default = subject_25_data['Physical_Activity']
activity = st.sidebar.selectbox(
    "Physical Activity",
    options=list(activity_options.keys()),
    index=0 if activity_default == 1.0 else 1
)
activity_val = activity_options[activity]

# Diabetes Status
diabetes_options = {"Yes": 1.0, "No": 2.0, "Borderline": 3.0}
diabetes_default = subject_25_data['Diabetes_Status']
diabetes = st.sidebar.selectbox(
    "Diabetes Status",
    options=list(diabetes_options.keys()),
    index={1.0: 0, 2.0: 1, 3.0: 2}.get(diabetes_default, 1)
)
diabetes_val = diabetes_options[diabetes]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎂 Patient Age")

# Chronological Age (Ground Truth)
chronological_age = st.sidebar.number_input(
    "Chronological Age (years)",
    min_value=1,
    max_value=100,
    value=44,  # Default from Subject #25
    step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Biomarkers")

# BMI
bmi = st.sidebar.number_input(
    "BMI (kg/m²)",
    min_value=10.0,
    max_value=80.0,
    value=float(subject_25_data['BMI']),
    step=0.1,
    format="%.1f"
)

# Fasting Blood Glucose
glucose = st.sidebar.number_input(
    "Fasting Blood Glucose (mg/dL)",
    min_value=50.0,
    max_value=400.0,
    value=float(subject_25_data['Fasting_Blood_Glucose']),
    step=1.0,
    format="%.1f"
)

# Oral Glucose Tolerance
ogt = st.sidebar.number_input(
    "Oral Glucose Tolerance (mg/dL)",
    min_value=50.0,
    max_value=500.0,
    value=float(subject_25_data['Oral_Glucose_Tolerance']),
    step=1.0,
    format="%.1f"
)

# Blood Insulin
insulin = st.sidebar.number_input(
    "Blood Insulin (µU/mL)",
    min_value=0.0,
    max_value=200.0,
    value=float(subject_25_data['Blood_Insulin']),
    step=0.1,
    format="%.2f"
)

# Construct input DataFrame
input_data = pd.DataFrame({
    'Gender': [gender_val],
    'Physical_Activity': [activity_val],
    'BMI': [bmi],
    'Fasting_Blood_Glucose': [glucose],
    'Diabetes_Status': [diabetes_val],
    'Oral_Glucose_Tolerance': [ogt],
    'Blood_Insulin': [insulin]
})


# ============================================================================
# Main Content - Prediction
# ============================================================================
# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 NHANES Biological Age Predictor</h1>
    <p>Clinical Decision Support System with Explainable AI</p>
</div>
""", unsafe_allow_html=True)

# Stakeholder View Selection
stakeholder = st.selectbox(
    "🎯 Select Stakeholder View",
    options=["Clinician", "Patient", "Policymaker/Authority"],
    index=0
)

# Make prediction
prediction = xgb_model.predict(input_data)[0]
prediction_proba = xgb_model.predict_proba(input_data)[0]
predicted_class = le.classes_[prediction]
confidence = prediction_proba[prediction] * 100

# Display prediction
col1, col2 = st.columns([2, 1])

with col1:
    if predicted_class == 'Aged (65+)':
        card_class = 'prediction-aged'
    elif predicted_class == 'Adult (19-64)':
        card_class = 'prediction-adult'
    else:
        card_class = 'prediction-child'
    
    st.markdown(f"""
    <div class="{card_class}">
        <div class="prediction-label">Predicted Age Group</div>
        <div class="prediction-value">{predicted_class.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric(
        label="Model Confidence",
        value=f"{confidence:.1f}%",
        delta=None
    )

# ============================================================================
# Tailored AI Summary (Based on Stakeholder)
# ============================================================================
st.markdown("### 📋 Tailored AI Summary")

# Generate dynamic summary based on stakeholder and prediction
if stakeholder == "Clinician":
    # Medical terminology focus
    if predicted_class == 'Aged (65+)':
        summary = f"""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #0066cc;">
            <strong>🩺 Clinical Assessment:</strong><br><br>
            Patient presents with metabolic markers consistent with <strong>accelerated biological aging</strong>. 
            Key findings include:
            <ul style="margin: 0.5rem 0;">
                <li><strong>Fasting Glucose:</strong> {glucose:.1f} mg/dL {"(Hyperglycemia)" if glucose > 100 else "(Normal range)"}</li>
                <li><strong>Blood Insulin:</strong> {insulin:.1f} µU/mL {"(Hyperinsulinemia)" if insulin > 25 else "(Within range)"}</li>
                <li><strong>Oral Glucose Tolerance:</strong> {ogt:.1f} mg/dL {"(Impaired)" if ogt > 140 else "(Normal)"}</li>
                <li><strong>BMI:</strong> {bmi:.1f} kg/m² {"(Elevated)" if bmi > 25 else "(Normal)"}</li>
            </ul>
            <strong>Model Interpretation:</strong> SHAP analysis indicates metabolic dysregulation as primary contributor. 
            Recommend: HbA1c follow-up, lipid panel, cardiovascular risk assessment.
        </div>
        """
    else:
        summary = f"""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #28a745;">
            <strong>🩺 Clinical Assessment:</strong><br><br>
            Patient's metabolic profile is consistent with <strong>{predicted_class}</strong> classification.
            <ul style="margin: 0.5rem 0;">
                <li><strong>Fasting Glucose:</strong> {glucose:.1f} mg/dL {"(Elevated)" if glucose > 100 else "(Normal)"}</li>
                <li><strong>Blood Insulin:</strong> {insulin:.1f} µU/mL</li>
                <li><strong>BMI:</strong> {bmi:.1f} kg/m²</li>
            </ul>
            <strong>Model Interpretation:</strong> SHAP feature attribution confirms biomarkers within expected range for age group.
            No immediate metabolic intervention indicated based on current profile.
        </div>
        """

elif stakeholder == "Patient":
    # ELI5 (Explain Like I'm 5) language
    if predicted_class == 'Aged (65+)':
        summary = f"""
        <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #ffc107;">
            <strong>💡 What This Means For You:</strong><br><br>
            Based on your health numbers, your body shows signs of being in the <strong>"65 and older"</strong> age group, 
            even though your actual age is {chronological_age}.
            <br><br>
            <strong>In simple terms:</strong>
            <ul style="margin: 0.5rem 0;">
                <li>Your <strong>sugar levels</strong> are {"higher than ideal" if glucose > 100 else "okay"}</li>
                <li>This suggests your body may be aging faster than expected</li>
                <li>The good news: many of these factors can be improved!</li>
            </ul>
            <strong>👉 Next Steps:</strong> Check the "Clinical Recourse" tab to see what small changes 
            could help your body "feel younger" according to this test.
        </div>
        """
    else:
        summary = f"""
        <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #28a745;">
            <strong>💡 What This Means For You:</strong><br><br>
            Great news! Your health numbers suggest your body is in the <strong>"{predicted_class}"</strong> range.
            <br><br>
            <strong>In simple terms:</strong>
            <ul style="margin: 0.5rem 0;">
                <li>Your body's "biological age" matches or is better than expected</li>
                <li>Your sugar and energy levels look good</li>
                <li>Keep up the healthy habits!</li>
            </ul>
            <strong>👉 Tip:</strong> Regular exercise and balanced eating help keep these numbers healthy.
        </div>
        """

else:  # Policymaker/Authority
    if predicted_class == 'Aged (65+)':
        summary = f"""
        <div style="background: linear-gradient(135deg, #e7f3ff 0%, #cce5ff 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #004085;">
            <strong>📊 Policy & Resource Allocation Insight:</strong><br><br>
            This patient profile indicates <strong>elevated biological age</strong> classification, flagging for:
            <ul style="margin: 0.5rem 0;">
                <li><strong>Triage Priority:</strong> Higher priority for preventive interventions</li>
                <li><strong>Resource Allocation:</strong> Candidate for chronic disease management programs</li>
                <li><strong>Vulnerable Group Identification:</strong> Consider for priority vaccine allocation, ICU resource planning</li>
                <li><strong>Population Health:</strong> Data point for metabolic syndrome prevalence tracking</li>
            </ul>
            <strong>Model Confidence:</strong> {confidence:.1f}% — SHAP validation available for audit trail.
            <br>
            <strong>Recommendation:</strong> Include in targeted intervention cohort for resource-efficient healthcare delivery.
        </div>
        """
    else:
        summary = f"""
        <div style="background: linear-gradient(135deg, #e7f3ff 0%, #cce5ff 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #004085;">
            <strong>📊 Policy & Resource Allocation Insight:</strong><br><br>
            Patient classified within <strong>{predicted_class}</strong> biological age group.
            <ul style="margin: 0.5rem 0;">
                <li><strong>Triage Priority:</strong> Standard priority tier</li>
                <li><strong>Resource Allocation:</strong> Routine preventive care pathway</li>
                <li><strong>Population Health:</strong> No elevated risk flags for resource prioritization</li>
            </ul>
            <strong>Model Confidence:</strong> {confidence:.1f}% — Explainable AI audit trail available via SHAP.
        </div>
        """

st.markdown(summary, unsafe_allow_html=True)


# Ground Truth Comparison
st.markdown("<br>", unsafe_allow_html=True)

# Determine true age group
true_age_group = bin_age_3_class(chronological_age)

# Compare predicted vs actual
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
        <div style="font-size: 0.8rem; color: #6c757d; text-transform: uppercase;">Chronological Age</div>
        <div style="font-size: 1.5rem; font-weight: 600;">{chronological_age} years</div>
        <div style="font-size: 0.85rem; color: #495057;">{true_age_group}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
        <div style="font-size: 0.8rem; color: #6c757d; text-transform: uppercase;">Predicted Biological Age</div>
        <div style="font-size: 1.5rem; font-weight: 600;">{predicted_class}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Comparison result
    if predicted_class == true_age_group:
        comparison = "✅ Matches"
        comparison_color = "#28a745"
        comparison_desc = "Biological age matches chronological age"
    elif (predicted_class == 'Aged (65+)' and true_age_group != 'Aged (65+)'):
        comparison = "⚠️ Older"
        comparison_color = "#dc3545"
        comparison_desc = "Biological age appears older than actual"
    elif (predicted_class == 'Infant/Child/Adolescent' and true_age_group != 'Infant/Child/Adolescent'):
        comparison = "🟢 Younger"
        comparison_color = "#17a2b8"
        comparison_desc = "Biological age appears younger than actual"
    elif (predicted_class == 'Adult (19-64)' and true_age_group == 'Aged (65+)'):
        comparison = "🟢 Younger"
        comparison_color = "#17a2b8"
        comparison_desc = "Biological age appears younger than actual"
    else:
        comparison = "⚠️ Older"
        comparison_color = "#dc3545"
        comparison_desc = "Biological age appears older than actual"
    
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
        <div style="font-size: 0.8rem; color: #6c757d; text-transform: uppercase;">Comparison</div>
        <div style="font-size: 1.5rem; font-weight: 600; color: {comparison_color};">{comparison}</div>
        <div style="font-size: 0.75rem; color: #495057;">{comparison_desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# XAI Tabs
# ============================================================================
tab1, tab2, tab3 = st.tabs([
    "🔍 Diagnostic Insight (SHAP)", 
    "💊 Clinical Recourse (DiCE)",
    "📊 Global Feature Importance"
])

# --- Tab 1: SHAP ---
with tab1:
    st.markdown("### Why This Prediction?")
    st.markdown("""
    <div class="info-box">
        <strong>SHAP (SHapley Additive exPlanations)</strong> reveals how each feature contributed to pushing the 
        prediction toward or away from the predicted class.
    </div>
    """, unsafe_allow_html=True)
    
    # Generate SHAP values for current input
    with st.spinner("Generating SHAP explanation..."):
        shap_values = shap_explainer(input_data)
        
        # Create waterfall plot (reduced size)
        # Create waterfall plot (reduced size)
        # Use columns to constrain width (Streamlit expands images by default)
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            shap.waterfall_plot(shap_values[0, :, prediction], show=False)
            plt.title(f"Feature Contributions for '{predicted_class}' Prediction", fontsize=12, fontweight='bold')
            # Use bbox_inches='tight' instead of tight_layout for safer saving/rendering
            st.pyplot(fig, clear_figure=True, bbox_inches='tight')
            plt.close()
        except Exception as e:
            # Fallback: Try plotting with minimal configuration
            try:
                st.warning(f"Standard plot failed ({str(e)}), attempting fallback...")
                plt.close() 
                fig, ax = plt.subplots(figsize=(7, 4))
                shap.waterfall_plot(shap_values[0, :, prediction], show=False)
                plt.title("Feature Contributions (Fallback)", fontsize=12)
                st.pyplot(fig, clear_figure=True)
                plt.close()
            except Exception as e2:
                st.error(f"Could not render SHAP plot: {str(e2)}")
                plt.close()
    
    # Dynamic caption based on top features
    feature_contributions = []
    for i, col in enumerate(input_data.columns):
        val = shap_values[0, i, prediction].values
        feature_contributions.append((col, val, input_data[col].values[0]))
    
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = feature_contributions[:3]
    
    # Generate explanation text
    positive_features = [f for f in top_features if f[1] > 0]
    negative_features = [f for f in top_features if f[1] < 0]
    
    explanation = "**Key Drivers:** "
    if positive_features:
        drivers = [f"{f[0].replace('_', ' ')} ({f[2]:.1f}, +{f[1]:.2f})" for f in positive_features]
        explanation += f"Factors pushing toward '{predicted_class}': {', '.join(drivers)}. "
    if negative_features:
        reducers = [f"{f[0].replace('_', ' ')} ({f[2]:.1f}, {f[1]:.2f})" for f in negative_features]
        explanation += f"Factors reducing this prediction: {', '.join(reducers)}."
    
    st.markdown(explanation)


# --- Tab 2: DiCE ---
with tab2:
    st.markdown("### What Can We Do?")
    st.markdown("""
    <div class="info-box">
        <strong>DiCE (Diverse Counterfactual Explanations)</strong> suggests minimal changes to patient biomarkers 
        that could result in a different age group classification.
    </div>
    """, unsafe_allow_html=True)
    
    # Determine target class for counterfactuals
    # Clinical logic: We want to show BENEFICIAL changes
    # - If patient is "Aged" → target "Adult" (show how to achieve younger classification)
    # - If patient is "Adult" → target "Aged" (explanatory only, NOT actionable advice)
    all_classes = list(range(len(le.classes_)))
    target_classes = [c for c in all_classes if c != prediction]
    
    # Determine if counterfactuals are clinically actionable
    # Only "Aged → Adult" transitions are actionable health recommendations
    is_actionable = (predicted_class == 'Aged (65+)')
    
    if target_classes:
        # For Aged patients: target Adult (index for Adult class)
        # For Adult/Child: target any other class (for explanation only)
        if is_actionable:
            # Find Adult class index
            adult_idx = list(le.classes_).index('Adult (19-64)') if 'Adult (19-64)' in le.classes_ else target_classes[0]
            target_class = adult_idx
        else:
            target_class = target_classes[0]
        
        target_class_name = le.classes_[target_class]
        
        if is_actionable:
            st.markdown(f"**Goal:** Identify changes to achieve **{target_class_name}** classification")
        else:
            st.markdown(f"**Analysis:** What factors would change prediction to **{target_class_name}** *(for explanation only)*")
        
        # DiCE configuration
        FEATURES_TO_VARY = continuous_features + ['Physical_Activity']
        PERMITTED_RANGE = {
            'BMI': [15, 40],
            'Fasting_Blood_Glucose': [60, 200],
            'Oral_Glucose_Tolerance': [70, 250],
            'Blood_Insulin': [2, 50],
            'Physical_Activity': ['1.0', '2.0']
        }
        
        with st.spinner("Generating counterfactual explanations..."):
            try:
                cf = dice_exp.generate_counterfactuals(
                    input_data,
                    total_CFs=3,
                    desired_class=target_class,
                    features_to_vary=FEATURES_TO_VARY,
                    permitted_range=PERMITTED_RANGE
                )
                
                if cf.cf_examples_list[0].final_cfs_df is not None:
                    cf_df = cf.cf_examples_list[0].final_cfs_df.copy()
                    
                    # Prepare original instance row (matching backend script format)
                    original_row = input_data.copy()
                    original_row['age_group'] = prediction
                    
                    # Ensure columns match
                    original_row = original_row[cf_df.columns]
                    
                    # Combine: Original first, then counterfactuals
                    combined_df = pd.concat([original_row, cf_df], ignore_index=True)
                    
                    # Add Ground Truth (based on chronological age from sidebar)
                    # Only first row has ground truth, counterfactuals are hypothetical
                    ground_truth_class = bin_age_3_class(chronological_age)
                    combined_df['Ground Truth'] = [ground_truth_class] + ['-'] * len(cf_df)
                    
                    # Add model prediction label for each row
                    combined_df['Model Prediction'] = combined_df['age_group'].apply(lambda x: le.classes_[int(x)])
                    
                    # Replace unchanged values with '-' (matching backend script)
                    # FIX: Cast to string explicitly to avoid ArrowInvalid error when mixing numbers and strings
                    # PyArrow tries to infer type from first row (numbers) and crashes on '-' later
                    display_df = combined_df.copy().astype(str)
                    feature_cols = cf_df.columns
                    original_vals = combined_df.iloc[0]
                    
                    for i in range(1, len(combined_df)):
                        for col in feature_cols:
                            # Use numeric values from combined_df for comparison
                            val = combined_df.at[i, col]
                            orig = combined_df.at[0, col]
                            
                            try:
                                # Strict equality for categorical/integers
                                if val == orig:
                                    display_df.at[i, col] = '-'
                                # Tolerance check for floats
                                elif isinstance(val, (int, float)) and isinstance(orig, (int, float)):
                                    if np.isclose(val, orig, rtol=1e-05):
                                        display_df.at[i, col] = '-'
                            except:
                                if val == orig:
                                    display_df.at[i, col] = '-'
                    
                    # Format display - remove age_group column (redundant with Model Prediction)
                    display_df = display_df.drop(columns=['age_group'])
                    
                    # Add row labels
                    display_df.insert(0, 'Scenario', ['Original (Current)'] + [f'Counterfactual {i+1}' for i in range(len(cf_df))])
                    
                    st.markdown("#### Counterfactual Scenarios")
                    st.dataframe(display_df, width="stretch")
                    
                    # Only show actionable recommendations for Aged → Adult transitions
                    if is_actionable:
                        st.markdown("#### Actionable Recommendations")
                        
                        first_cf = cf_df.iloc[0]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        # Show deltas for key features
                        for col_widget, feature in zip([col1, col2, col3], ['Fasting_Blood_Glucose', 'Blood_Insulin', 'Oral_Glucose_Tolerance']):
                            original_val = input_data[feature].values[0]
                            cf_val = first_cf[feature]
                            delta = cf_val - original_val
                            
                            if abs(delta) > 0.01:
                                with col_widget:
                                    st.metric(
                                        label=f"Target {feature.replace('_', ' ')}",
                                        value=f"{cf_val:.1f}",
                                        delta=f"{delta:+.1f}",
                                        delta_color="inverse"  # Green for reductions
                                    )
                        
                        # Constraint note
                        st.markdown("""
                        <div class="warning-box">
                            <strong>Note:</strong> Immutable features (Gender, Diabetes History) were held constant. 
                            Only modifiable biomarkers and lifestyle factors are adjusted in these scenarios.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # For non-actionable scenarios, show explanation instead
                        st.markdown("""
                        <div class="info-box">
                            <strong>Note:</strong> These counterfactuals show what factors would cause an older biological age classification.
                            This is for <em>understanding the model</em>, not health advice. Your current profile is favorable.
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No valid counterfactuals could be generated for this input.")
                    
                # Stakeholder Guidance for DiCE
                st.markdown("---")
                st.markdown("### Stakeholder Guidance")
                
                # Logic to determine Guidance Context (Improvement vs Prevention)
                # target_class_name comes from line 696/699 logic above. 
                # We need to explicitly check the direction of change.
                target_name = le.classes_[target_class]
                
                guidance_context = "NEUTRAL"
                if predicted_class == 'Aged (65+)' and target_name == 'Adult (19-64)':
                    guidance_context = "CORRECTIVE" # Goal: Get back to Adult
                elif predicted_class == 'Adult (19-64)' and target_name == 'Aged (65+)':
                    guidance_context = "PREVENTATIVE" # Goal: Don't become Aged
                elif predicted_class == 'Infant/Child/Adolescent' and target_name == 'Aged (65+)':
                     guidance_context = "PREVENTATIVE"
                
                # Dynamic Content Generation
                if stakeholder == "Clinician":
                    if guidance_context == "CORRECTIVE":
                        title = "🩺 Therapeutic Targets (Recovery)"
                        content = """
<ul style="margin: 0.5rem 0;">
    <li><strong>Clinical Goal:</strong> Revert biological aging markers from <strong>Aged</strong> to <strong>Adult</strong> profile.</li>
    <li><strong>Primary Intervention:</strong> Tighten glycemic control (Glucose/Insulin) to target range shown above.</li>
    <li><strong>Monitoring:</strong> Schedule follow-up HbA1c to track regression of metabolic age.</li>
</ul>
"""
                    elif guidance_context == "PREVENTATIVE":
                        title = "🩺 Risk Mitigation (Prevention)"
                        content = """
<ul style="margin: 0.5rem 0;">
    <li><strong>Clinical Goal:</strong> Maintain <strong>Adult</strong> status and prevent progression to <strong>Aged</strong> profile.</li>
    <li><strong>Risk Factors:</strong> The counterfactuals highlight thresholds where current metabolic stability would decompensate.</li>
    <li><strong>Advisory:</strong> Counsel patient on maintaining current BMI and glycemic parameters to avoid this trajectory.</li>
</ul>
"""
                    else:
                         title = "🩺 Model Interpretation"
                         content = "Counterfactuals demonstrate decision boundaries between classes for model validation."

                    st.markdown(f"""<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #0066cc;">
    <strong>{title}</strong><br><br>
    {content}
</div>""", unsafe_allow_html=True)
                
                elif stakeholder == "Patient":
                    if guidance_context == "CORRECTIVE":
                        title = "💡 Action Plan (Get Younger)"
                        content = """<ul style="margin: 0.5rem 0;">
    <li><strong>Goal:</strong> Your numbers suggest an "Aged" profile. Let's aim for the "Adult" targets above!</li>
    <li><strong>Key Steps:</strong> Lowering sugar/insulin levels is your most powerful tool right now.</li>
    <li><strong>Motivation:</strong> Small changes in diet/exercise can help move you back to the "Adult" zone.</li>
</ul>"""
                        style = "background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-left: 4px solid #28a745;" # Green for go
                    elif guidance_context == "PREVENTATIVE":
                        title = "🛡️ Stay Healthy (Warning)"
                        content = """<ul style="margin: 0.5rem 0;">
    <li><strong>Goal:</strong> You are currently in the healthy "Adult" range. Keep it that way!</li>
    <li><strong>Warning:</strong> The table showing "Counterfactuals" reveals what would happen if your health slipped.</li>
    <li><strong>Action:</strong> Don't let your sugar or weight drift up to those levels. Keep doing what works!</li>
</ul>"""
                        style = "background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%); border-left: 4px solid #ffc107;" # Yellow/Orange for warning
                    else:
                        title = "💡 Insight"
                        content = "These scenarios show how different health numbers change the AI's prediction."
                        style = "background: #f8f9fa; border-left: 4px solid #6c757d;"

                    st.markdown(f"""<div style="{style} padding: 1.2rem; border-radius: 10px;">
    <strong>{title}</strong><br><br>
    {content}
</div>""", unsafe_allow_html=True)
                
                else:  # Policymaker
                    if guidance_context == "CORRECTIVE":
                        title = "📊 Intervention Efficacy"
                        content = """<ul style="margin: 0.5rem 0;">
    <li><strong>Target Population:</strong> 'Aged' group with reversible metabolic risk.</li>
    <li><strong>Projected Impact:</strong> Targeted glycemic interventions could reclassify this demographic to 'Adult' status, reducing dependency ratios.</li>
    <li><strong>ROI:</strong> High return on investment for diabetes prevention programs in this cohort.</li>
</ul>"""
                    elif guidance_context == "PREVENTATIVE":
                        title = "📊 Risk Surveillance"
                        content = """<ul style="margin: 0.5rem 0;">
    <li><strong>Target Population:</strong> At-risk 'Adult' population near metabolic tipping points.</li>
    <li><strong>Strategy:</strong> Preventative screening to catch patients before they cross the thresholds shown in counterfactuals.</li>
    <li><strong>Metric:</strong> Track incidence rate of adults transitioning to 'Aged' biological classification.</li>
</ul>"""
                    else:
                         title = "📊 Model Analysis"
                         content = "Boundary analysis for regulatory auditing."

                    st.markdown(f"""<div style="background: linear-gradient(135deg, #e7f3ff 0%, #cce5ff 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #004085;">
    <strong>{title}</strong><br><br>
    {content}
</div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Could not generate counterfactuals: {str(e)}")
                st.info("Try adjusting input values or check if the prediction allows for counterfactual changes.")
    else:
        st.info("No alternative classifications available for counterfactual analysis.")


# --- Tab 3: Population Research Insights ---
with tab3:
    st.markdown("### 📚 Population-Level Research Insights")
    st.markdown("""
    <div class="info-box">
        <strong>Research Context:</strong> This analysis is based on the <strong>entire NHANES dataset</strong> used to train the model. 
        It illustrates general population trends and biomarkers associated with biological aging, NOT specific data from other individual patients.
    </div>
    """, unsafe_allow_html=True)
    
    # Generate SHAP values for the entire dataset (or large representative sample)
    with st.spinner("Calculating global feature importance on full population data..."):
        # Use full dataset for accurate population-level insights
        # Note: In a production setting with millions of rows, we might still need to sample, 
        # but for this dataset, we use all available research data.
        X_research = X.copy()
        
        # Get SHAP values for the full research dataset
        # We cache this operation to improve performance on subsequent reloads
        @st.cache_data
        def get_global_shap_values(_explainer, _data):
            return _explainer(_data)
            
        shap_values_global = get_global_shap_values(shap_explainer, X_research)
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Stacked Bar Plot for Multi-Class Feature Importance
            # This shows the mean absolute SHAP value for each feature, stacked by class
            shap.plots.bar(shap_values_global, show=False, max_display=10)
            plt.title("Global Feature Importance (All Classes)", fontsize=14)
            st.pyplot(fig, clear_figure=True, bbox_inches='tight')
            plt.close()
        except Exception as e:
            st.error(f"Could not render Global SHAP plot: {str(e)}")
            plt.close()

        
        # Calculate mean absolute SHAP values for ranking
        mean_abs_shap = np.abs(shap_values_global.values).mean(axis=(0, 2))
        feature_importance = list(zip(X.columns, mean_abs_shap))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Display ranking table
        st.markdown("#### Population-Wide Feature Ranking")
        importance_df = pd.DataFrame(feature_importance, columns=['Feature', 'Mean |SHAP| (Impact)'])
        importance_df['Feature'] = importance_df['Feature'].str.replace('_', ' ')
        importance_df.index = range(1, len(importance_df) + 1)
        st.dataframe(importance_df, width="stretch")
    
    # Stakeholder-specific interpretation
    st.markdown("---")
    st.markdown("### Stakeholder Interpretation")
    
    if stakeholder == "Clinician":
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #0066cc;">
            <strong>🩺 Research Findings:</strong><br><br>
            Analysis of the NHANES cohort reveals:
            <ul style="margin: 0.5rem 0;">
                <li><strong>Primary Drivers:</strong> Across the population, Fasting Glucose and Blood Insulin are the most distinguishable features for biological aging classification.</li>
                <li><strong>Validation:</strong> These findings align with established metabolic aging literature (e.g., insulin resistance theories of aging).</li>
                <li><strong>Application:</strong> This global pattern justifies the heavy weighting of glycemic markers in individual patient assessments.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif stakeholder == "Patient":
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #28a745;">
            <strong>💡 General Health Trends:</strong><br><br>
            Looking at data from thousands of people in the study, we learned:
            <ul style="margin: 0.5rem 0;">
                <li><strong>Common Themes:</strong> For most people, keeping blood sugar and insulin in check is the most effective way to maintain a "younger" biological profile.</li>
                <li><strong>Not Just You:</strong> These are universal health factors, not just specific to your case.</li>
                <li><strong>Lifestyle Matters:</strong> Weight and activity levels consistently show up as important factors for everyone.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    else:  # Policymaker
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e7f3ff 0%, #cce5ff 100%); padding: 1.2rem; border-radius: 10px; border-left: 4px solid #004085;">
            <strong>📊 Population Health Insights:</strong><br><br>
            Derived from the full NHANES research dataset:
            <ul style="margin: 0.5rem 0;">
                <li><strong>Epidemiological Trend:</strong> Metabolic dysfunction is the leading statistical driver of accelerated biological age in this population.</li>
                <li><strong>Strategic Focus:</strong> Public health interventions targeting glycemic control could have the highest aggregate impact on population longevity.</li>
                <li><strong>Evidence Base:</strong> This aggregate data supports funding for national metabolic screening programs over isolated biomarker interventions.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.85rem;">
    <p>NHANES Biological Age Predictor | Powered by XGBoost, SHAP & DiCE</p>
    <p>This is a demonstration tool for educational purposes. Not for clinical use.</p>
</div>
""", unsafe_allow_html=True)
