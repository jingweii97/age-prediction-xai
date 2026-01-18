# %% [markdown]
# # Data and Library Loading

# %%
# pip install ucimlrepo

# %%
# Snippet from https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+%28nhanes%29+age+prediction+subset
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
national_health_and_nutrition_health_survey_2013_2014_nhanes_age_prediction_subset = fetch_ucirepo(id=887) 

# data (as pandas dataframes) 
X = national_health_and_nutrition_health_survey_2013_2014_nhanes_age_prediction_subset.data.features
# We use age as target variable instead of age group
import pandas as pd
y = pd.DataFrame(national_health_and_nutrition_health_survey_2013_2014_nhanes_age_prediction_subset.data.original['RIDAGEYR'])
  
# metadata 
print(national_health_and_nutrition_health_survey_2013_2014_nhanes_age_prediction_subset.metadata) 
  
# variable information 
print(national_health_and_nutrition_health_survey_2013_2014_nhanes_age_prediction_subset.variables) 

# %%
# Rename to more intuitive names:
feature_name_dict = {
    "SEQN": "ID",
    "age_group": "Age_Group",
    "RIDAGEYR": "Age",
    "RIAGENDR": "Gender",
    "PAQ605": "Physical_Activity",
    "BMXBMI": "BMI",
    "LBXGLU": "Fasting_Blood_Glucose",
    "DIQ010": "Diabetes_Status",
    "LBXGLT": "Oral_Glucose_Tolerance",
    "LBXIN": "Blood_Insulin"
}
X = X.rename(columns=feature_name_dict)
y = y.rename(columns=feature_name_dict)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %% [markdown]
# # Data Integrity Check

# %%
print(f'Missing data in feature \n {X.isnull().sum()} \n in target:\n {y.isnull().sum()}')
print(f'Duplicate instance {X.duplicated().sum()}')

# %% [markdown]
# # Feature Engineering (Binning)

# %%


def bin_age_3_class(age):
    # Merging Adult and Middle Aged into one large "Adult" group
    if age <= 18:
        return 'Infant/Child/Adolescent'
    elif age <= 64:
        return 'Adult (19-64)'
    else:
        return 'Aged (65+)'


# Apply binning 
y_binned = y['Age'].apply(bin_age_3_class)
print("Using 3-CLASS Classification (Child / Adult / Aged)")

print("Target distribution after binning:")
print(y_binned.value_counts())

# Encode the categorical labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_binned)
print("\nEncoded classes:", le.classes_)

# %% [markdown]
# # EDA

# %%
X.shape

# %%
y_binned.shape

# %%
X.hist(bins=20, figsize=(15,10))
plt.show()

# %%
pd.Series(y_encoded).hist()
plt.show()

# %%
X.info()

# %% [markdown]
# # Data Preprocessing

# %%
# Use stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Training set class distribution:")
print(pd.Series(y_train).value_counts())
print("\nTest set class distribution:")
print(pd.Series(y_test).value_counts())

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
X_train.head()

# %% [markdown]
# # Model training with Class Weights

# %%
# Calculate class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print("Class weights:", class_weight_dict)

# %%

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
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

# Calculate sample weights for class imbalance
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

# Standard balanced weights
balanced_weights = compute_sample_weight('balanced', y_train)

# Soften the weights (Square Root)
# This reduces the penalty on the majority class while still helping minority classes
# Example: Instead of weights [0.3, 3.0], we get [0.55, 1.73]
soft_weights = np.sqrt(balanced_weights)

# Fit the model with softened weights
xgb_model.fit(X_train, y_train, sample_weight=soft_weights)

# %%
from sklearn.model_selection import cross_val_score

# Perform stratified 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

print("=" * 80)
print("CROSS-VALIDATION RESULTS (XGBoost)")
print("=" * 80)
print(f"CV Accuracy scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print("=" * 80)

# %%
# Predictions using XGBoost
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# %%
# Calculate metrics for training set
train_acc = accuracy_score(y_train, y_train_pred)

# Calculate metrics for test set
test_acc = accuracy_score(y_test, y_test_pred)

# Display results
print("=" * 80)
print("MODEL PERFORMANCE METRICS (XGBoost)")
print("=" * 80)
print(f"\n{'Metric':<25} {'Training Set':<20} {'Test Set':<20}")
print("-" * 80)
print(f"{'Accuracy':<25} {train_acc:<20.4f} {test_acc:<20.4f}")
print("=" * 80)

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# %%
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=le.classes_, yticklabels=le.classes_,
            annot_kws={'size': 18}) # Increased font size for numbers
plt.xlabel('Predicted Label', fontsize=18)
plt.ylabel('True Label', fontsize=18)
plt.title('Confusion Matrix (XGBoost)', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# %% [markdown]
# # Feature Importance Analysis (SHAP)

# %%
# pip install shap

# %%
import shap

# For multi-class XGBoost, we need to use a workaround due to SHAP compatibility issues
# Option 1: Use shap.Explainer with the predict function (works for all models)

# Create a background dataset (sample from training data for speed)
background = shap.sample(X_train, 100)

# Use Explainer with model's predict_proba for multi-class
explainer = shap.Explainer(xgb_model.predict_proba, background)

# Calculate SHAP values for a subset of test data (full set can be slow)
X_test_sample = X_test.iloc[:50]  # Use first 50 samples for speed
shap_values = explainer(X_test_sample)

print(f"SHAP values shape: {shap_values.shape}")
print(f"Classes: {le.classes_}")

# %% [markdown]
# ## Global Explanation: Beeswarm Summary Plot

# %%
# Beeswarm plot showing feature importance across all classes
# For multi-class, we show the mean absolute SHAP values
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", class_names=le.classes_, show=False)
plt.title("Global Feature Importance (SHAP - All Classes)")
plt.tight_layout()
plt.show()

# %%
# Detailed Beeswarm for each class
for i, class_name in enumerate(le.classes_):
    print(f"\n{'='*60}")
    print(f"SHAP Summary for Class: {class_name}")
    print(f"{'='*60}")
    plt.figure(figsize=(10, 6))
    # Extract SHAP values for this specific class
    shap.summary_plot(shap_values[:, :, i], X_test_sample, show=False)
    plt.title(f"Feature Impact on Predicting '{class_name}'")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Local Explanation: Waterfall Plot for a Single Instance

# %%
# Select a single instance from X_test for local explanation
instance_idx = 0  # Change this to explain different instances
single_instance = X_test.iloc[[instance_idx]]

# Get the prediction for this instance
predicted_class = xgb_model.predict(single_instance)[0]
predicted_class_name = le.classes_[predicted_class]
true_class_name = le.classes_[y_test[instance_idx]]

print(f"Instance {instance_idx}:")
print(f"  True Class: {true_class_name}")
print(f"  Predicted Class: {predicted_class_name}")
print(f"\nFeature Values:")
print(single_instance.T)

# %%
# Waterfall plot for the predicted class
# Get SHAP explanation for this single instance
single_shap = explainer(single_instance)

plt.figure(figsize=(10, 6))
# For waterfall, we need to create an Explanation object for the predicted class
shap.waterfall_plot(single_shap[0, :, predicted_class], show=False)
plt.title(f"Why the model predicted '{predicted_class_name}' for Instance {instance_idx}")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Counterfactual Explanations (DiCE)

# %%
# pip install dice-ml

# %%
import dice_ml
from dice_ml import Dice

# Prepare data for DiCE
# DiCE requires a DataFrame with both features and outcome
X_train_df = X_train.copy()
X_train_df['age_group'] = y_train

# Define continuous and categorical features
continuous_features = ['BMI', 'Fasting_Blood_Glucose', 'Oral_Glucose_Tolerance', 'Blood_Insulin']
categorical_features = ['Gender', 'Physical_Activity', 'Diabetes_Status']

# Create DiCE data object
dice_data = dice_ml.Data(
    dataframe=X_train_df,
    continuous_features=continuous_features,
    outcome_name='age_group'
)

# IMPORTANT: DiCE automatically treats any feature NOT in 'continuous_features' as CATEGORICAL
# Since Physical_Activity is in the dataframe but NOT in continuous_features,
# DiCE will treat it as categorical and only generate discrete values (1 or 2), not continuous

# %%
# Create a wrapper for the XGBoost model that DiCE can use
# DiCE needs a model that can predict on DataFrames

class XGBModelWrapper:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            # DiCE passes categorical features as strings (object dtype)
            # XGBoost expects numeric inputs (as it was trained on floats)
            # We must convert object columns back to numeric
            X_numeric = X[self.feature_names].apply(pd.to_numeric, errors='coerce')
            return self.model.predict(X_numeric)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            # Convert to numeric to handle string inputs from DiCE
            X_numeric = X[self.feature_names].apply(pd.to_numeric, errors='coerce')
            return self.model.predict_proba(X_numeric)
        return self.model.predict_proba(X)

# Wrap the model
feature_names = list(X_train.columns)
wrapped_model = XGBModelWrapper(xgb_model, feature_names)

# Create DiCE model object
dice_model = dice_ml.Model(model=wrapped_model, backend="sklearn", model_type="classifier")

# %%
# Initialize DiCE explainer
dice_exp = Dice(dice_data, dice_model, method="random")

# %%
# Global DiCE Configuration
# Define which features DiCE can vary and their permitted ranges

# Features that can be modified (continuous + actionable categorical)
FEATURES_TO_VARY_DICE = continuous_features + ['Physical_Activity']

# Permitted ranges for each feature
# Note: Physical_Activity is categorical (defined in dice_data), so DiCE will only
# generate discrete values that exist in the training data (1 or 2), not continuous values
PERMITTED_RANGE_DICE = {
    'BMI': [15, 40],
    'Fasting_Blood_Glucose': [60, 200],
    'Oral_Glucose_Tolerance': [70, 250],
    'Blood_Insulin': [2, 50],
    'Physical_Activity': ['1.0', '2.0']  # Must match string format of training data
}

# %%
# Select an instance from X_test# Generate counterfactuals for a specific instance
cf_instance_idx = 0  # Change this to explain different instances
query_instance = X_test.iloc[[cf_instance_idx]]

# Get the original prediction for this instance
original_pred = xgb_model.predict(query_instance)[0]
original_class = le.classes_[original_pred]

print(f"\nOriginal instance (index {cf_instance_idx}):")
print(query_instance)
print(f"\nOriginal prediction: {original_class}")

# For multi-class, we need to specify the desired_class
# Let's get all classes and pick a target class different from the original
all_classes = list(range(len(le.classes_)))
target_class = [c for c in all_classes if c != original_pred][0]  # Pick the first different class
target_class_name = le.classes_[target_class]

print(f"Target class for counterfactuals: {target_class_name}")

# Generate counterfactuals using global configuration
cf = dice_exp.generate_counterfactuals(
    query_instance,
    total_CFs=3,
    desired_class=target_class,
    features_to_vary=FEATURES_TO_VARY_DICE,
    permitted_range=PERMITTED_RANGE_DICE
)

# %%
# Display counterfactual results
print("\n" + "=" * 80)
print("COUNTERFACTUAL EXPLANATIONS")
print("=" * 80)
print(f"\nOriginal Prediction: {original_class}")
print("\nCounterfactuals (What-If Scenarios):")
cf.visualize_as_dataframe(show_only_changes=True)

# %%
# Get the counterfactuals as a DataFrame for further analysis
cf_df = cf.cf_examples_list[0].final_cfs_df
print("\nDetailed Counterfactual Table:")
print(cf_df)

# %% [markdown]
# # Export SHAP and DiCE Results for All Classes (TP/TN/FP/FN)

# %%
import os

# Create results directory structure
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Create subdirectories for each class and category
for class_name in le.classes_:
    safe_class_name = class_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    for category in ["TP", "TN", "FP", "FN"]:
        os.makedirs(f"{results_dir}/{safe_class_name}/{category}/SHAP", exist_ok=True)
        os.makedirs(f"{results_dir}/{safe_class_name}/{category}/DiCE", exist_ok=True)

print(f"Created results directory structure in '{results_dir}/'")

# %%
# Helper function to get indices for each category
def get_category_indices(y_true, y_pred, class_idx, n_samples=5):
    """
    Get indices for TP, TN, FP, FN for a specific class.
    
    For class C:
    - TP: True=C, Pred=C (correctly identified as C)
    - TN: True≠C, Pred≠C (correctly identified as NOT C)
    - FP: True≠C, Pred=C (wrongly predicted as C)
    - FN: True=C, Pred≠C (missed, should have been C)
    """
    tp_mask = (y_true == class_idx) & (y_pred == class_idx)
    tn_mask = (y_true != class_idx) & (y_pred != class_idx)
    fp_mask = (y_true != class_idx) & (y_pred == class_idx)
    fn_mask = (y_true == class_idx) & (y_pred != class_idx)
    
    tp_indices = np.where(tp_mask)[0][:n_samples]
    tn_indices = np.where(tn_mask)[0][:n_samples]
    fp_indices = np.where(fp_mask)[0][:n_samples]
    fn_indices = np.where(fn_mask)[0][:n_samples]
    
    return {
        "TP": tp_indices,
        "TN": tn_indices,
        "FP": fp_indices,
        "FN": fn_indices
    }

# %%
# Function to generate and save SHAP waterfall plot
def save_shap_waterfall(instance_df, instance_idx, predicted_class, true_class, save_path, explainer, le):
    """Generate and save SHAP waterfall plot for a single instance."""
    try:
        shap_values = explainer(instance_df)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap_values[0, :, predicted_class], show=False)
        
        pred_name = le.classes_[predicted_class]
        true_name = le.classes_[true_class]
        plt.title(f"Instance {instance_idx} | True: {true_name} | Pred: {pred_name}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"  Warning: Could not generate SHAP for instance {instance_idx}: {e}")
        return False

# %%
# Function to generate and save DiCE counterfactuals
def save_dice_counterfactuals(instance_df, instance_idx, original_class, target_class, save_path, dice_exp, le, continuous_features):
    """Generate and save DiCE counterfactual table for a single instance."""
    try:
        # Generate counterfactuals using global configuration
        cf = dice_exp.generate_counterfactuals(
            instance_df,
            total_CFs=3,
            desired_class=target_class,
            features_to_vary=FEATURES_TO_VARY_DICE,
            permitted_range=PERMITTED_RANGE_DICE
        )
        
        # Get counterfactuals as DataFrame
        if cf.cf_examples_list[0].final_cfs_df is not None:
            cf_df = cf.cf_examples_list[0].final_cfs_df.copy()
            
            # Prepare the Original Instance Row
            # instance_df does not have the target column 'age_group', so we add it
            original_row = instance_df.copy()
            original_row['age_group'] = original_class # original_class is the integer class
            
            # Ensure columns match (DiCE might reorder)
            original_row = original_row[cf_df.columns]
            
            # Create a combined DataFrame
            # First row: Original
            # Subsequent rows: Counterfactuals
            combined_df = pd.concat([original_row, cf_df], ignore_index=True)
            
            # Add text labels for classes
            combined_df['original_class'] = le.classes_[original_class]
            combined_df['current_prediction'] = combined_df['age_group'].apply(lambda x: le.classes_[int(x)])
            
            # Calculate changes (Diff)
            # We want to keep the first row (Original) as is.
            # For subsequent rows, if a value matches the original, we replace it with '-' or ''
            # We do this for feature columns only, not the metadata we just added.
            
            display_df = combined_df.copy()
            feature_cols = cf_df.columns # These are the features + target column used by DiCE
            
            original_vals = display_df.iloc[0]
            
            for i in range(1, len(display_df)):
                for col in feature_cols:
                    # Check for equality (handle floats with small tolerance if needed, but '==' usually works for equality check here as values are often copied)
                    # For continuous values from DiCE, they might differ slightly.
                    val = display_df.at[i, col]
                    orig = original_vals[col]
                    
                    try:
                        # Simple equality check
                        if val == orig:
                            display_df.at[i, col] = '-'
                        elif isinstance(val, (int, float)) and isinstance(orig, (int, float)):
                            if np.isclose(val, orig, rtol=1e-05):
                                display_df.at[i, col] = '-'
                    except:
                        if val == orig:
                            display_df.at[i, col] = '-'
            
            # Save to CSV
            display_df.to_csv(save_path, index=False)
            return True
        return False
    except Exception as e:
        print(f"  Warning: Could not generate DiCE for instance {instance_idx}: {e}")
        return False

# %%
# Main export loop
print("=" * 80)
print("EXPORTING SHAP AND DiCE RESULTS")
print("=" * 80)

n_instances = 20  # Number of instances per category

for class_idx, class_name in enumerate(le.classes_):
    safe_class_name = class_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    print(f"\n{'='*60}")
    print(f"Processing Class: {class_name} (index {class_idx})")
    print(f"{'='*60}")
    
    # Get indices for each category
    category_indices = get_category_indices(y_test, y_test_pred, class_idx, n_samples=n_instances)
    
    for category, indices in category_indices.items():
        print(f"\n  {category}: {len(indices)} instances found")
        
        for i, idx in enumerate(indices):
            instance_df = X_test.iloc[[idx]]
            true_class = y_test[idx]
            pred_class = y_test_pred[idx]
            
            # Save SHAP waterfall
            shap_path = f"{results_dir}/{safe_class_name}/{category}/SHAP/instance_{idx}.png"
            shap_success = save_shap_waterfall(
                instance_df, idx, pred_class, true_class, shap_path, explainer, le
            )
            
            # Save DiCE counterfactuals
            # For counterfactuals, we want to see what would change the prediction
            # Target: a class different from the predicted class
            target_classes = [c for c in range(len(le.classes_)) if c != pred_class]
            if target_classes:
                target_class = target_classes[0]
                dice_path = f"{results_dir}/{safe_class_name}/{category}/DiCE/instance_{idx}.csv"
                dice_success = save_dice_counterfactuals(
                    instance_df, idx, pred_class, target_class, dice_path, dice_exp, le, continuous_features
                )
            
            if shap_success:
                print(f"    ✓ Instance {idx}: SHAP saved")
            if dice_success:
                print(f"    ✓ Instance {idx}: DiCE saved")

print("\n" + "=" * 80)
print("EXPORT COMPLETE!")
print(f"Results saved to: {os.path.abspath(results_dir)}")
print("=" * 80)

# %%
# Generate summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

for class_idx, class_name in enumerate(le.classes_):
    safe_class_name = class_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    category_indices = get_category_indices(y_test, y_test_pred, class_idx, n_samples=100)
    
    print(f"\n{class_name}:")
    print(f"  TP (Correct {class_name}): {len(np.where((y_test == class_idx) & (y_test_pred == class_idx))[0])}")
    print(f"  TN (Correct NOT {class_name}): {len(np.where((y_test != class_idx) & (y_test_pred != class_idx))[0])}")
    print(f"  FP (False {class_name}): {len(np.where((y_test != class_idx) & (y_test_pred == class_idx))[0])}")
    print(f"  FN (Missed {class_name}): {len(np.where((y_test == class_idx) & (y_test_pred != class_idx))[0])}")

# %%
# Create a combined visualization grid for each class
print("\nGenerating combined visualizations...")

for class_idx, class_name in enumerate(le.classes_):
    safe_class_name = class_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Classification Results for: {class_name}", fontsize=16, fontweight='bold')
    
    categories = ["TP", "TN", "FP", "FN"]
    titles = [
        f"True Positive\n(Correctly predicted as {class_name})",
        f"True Negative\n(Correctly predicted as NOT {class_name})",
        f"False Positive\n(Wrongly predicted as {class_name})",
        f"False Negative\n(Missed, should be {class_name})"
    ]
    
    category_indices = get_category_indices(y_test, y_test_pred, class_idx, n_samples=100)
    
    for ax, cat, title in zip(axes.flat, categories, titles):
        count = len(category_indices[cat])
        ax.text(0.5, 0.5, f"{cat}\n\n{count} instances", 
                ha='center', va='center', fontsize=24, fontweight='bold',
                transform=ax.transAxes)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{safe_class_name}/summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary for {class_name}")

print("\nAll visualizations exported successfully!")
