"""
📂 Loading dataset...
   Loaded 11930 rows, 17 columns
✅ Dropped columns: []
🎯 Target: Target_2yr
   Class distribution:
Target_2yr
0.0    8894
1.0    3036
Name: count, dtype: int64
📝 Encoding categorical columns: ['PTGENDER']

📊 Feature columns (9):
['AGE', 'PTGENDER', 'PTEDUCAT', 'ADAS13', 'MOCA', 'CDRSB', 'FAQ', 'APOE4_count', 'GDTOTAL']

🔢 Numeric columns (7): ['AGE', 'PTEDUCAT', 'ADAS13', 'MOCA', 'CDRSB', 'FAQ', 'GDTOTAL']
📁 Categorical columns (2): ['PTGENDER', 'APOE4_count']

✂️ Train-test split:
   Training: 9544 samples
   Testing: 2386 samples

📈 Scaling numeric features...
✅ Scaling complete:
   X_train shape: (9544, 9)
   X_test shape: (2386, 9)
   Final column order: ['AGE', 'PTGENDER', 'PTEDUCAT', 'ADAS13', 'MOCA', 'CDRSB', 'FAQ', 'APOE4_count', 'GDTOTAL']

💾 Scaler saved to: models/progress_basic_scaler.pkl
💾 Feature metadata saved

⚖️ Applying SMOTE to balance classes...
   Before SMOTE: {0.0: 7115, 1.0: 2429}
   After SMOTE: {1.0: 7115, 0.0: 7115}
   Resampled training shape: (14230, 9)

🤖 Defining models...

🏋️ Training and evaluating models...

   Training LogisticRegression...
      ✓ Accuracy: 0.5935, ROC-AUC: 0.6603, F1: 0.4495

   Training RandomForest...
      ✓ Accuracy: 0.8265, ROC-AUC: 0.8834, F1: 0.6897

   Training GradientBoosting...
      ✓ Accuracy: 0.8093, ROC-AUC: 0.8658, F1: 0.6535

   Training XGBoost...
[14:29:39] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:738: 
Parameters: { "use_label_encoder" } are not used.

      ✓ Accuracy: 0.7536, ROC-AUC: 0.8765, F1: 0.6375

   Training LightGBM...
      ✓ Accuracy: 0.8294, ROC-AUC: 0.8759, F1: 0.6867

   Training SVM...
      ✓ Accuracy: 0.7959, ROC-AUC: 0.8347, F1: 0.6468

================================================================================
📋 MODEL COMPARISON (sorted by ROC-AUC)
================================================================================
             Model  Accuracy  ROC-AUC  F1-Score  Sensitivity  Specificity
      RandomForest    0.8265   0.8834    0.6897       0.7578       0.8499
           XGBoost    0.7536   0.8765    0.6375       0.8517       0.7201
          LightGBM    0.8294   0.8759    0.6867       0.7348       0.8617
  GradientBoosting    0.8093   0.8658    0.6535       0.7068       0.8443
               SVM    0.7959   0.8347    0.6468       0.7348       0.8168
LogisticRegression    0.5935   0.6603    0.4495       0.6524       0.5734
================================================================================

💾 Results saved to: models/progress_basic_model_comparison.csv

🏆 Best model: RandomForest
   ROC-AUC: 0.8834
   Accuracy: 0.8265
💾 Model saved to: models/progress_basic_RandomForest_best_model.pkl

🔍 Running SHAP analysis on RandomForest...
   Generating SHAP plots...
   💾 Feature importance plot saved: models/shap_feature_importance.png
No data for colormapping provided via 'c'. Parameters 'vmin', 'vmax' will be ignored
   💾 Summary plot saved: models/shap_summary.png
   ✅ SHAP analysis complete

================================================================================
✅ TRAINING PIPELINE COMPLETE
================================================================================
📊 Models trained: 6
🏆 Best model: RandomForest (ROC-AUC: 0.8834)
💾 Artifacts saved to: models/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 📂 STEP 1: LOAD DATASET
# ==========================================
print("📂 Loading dataset...")
df = pd.read_csv("data/ADNI_core_model_dataset.csv")
print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

# ==========================================
# 🧹 STEP 2: DROP UNWANTED COLUMNS
# ==========================================
drop_cols = ['RID', 'VISCODE', 'EXAMDATE', 'DIAGNOSIS', 'Target_1yr', 'Target_3yr', 'Target_5yr']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
print(f"✅ Dropped columns: {[c for c in drop_cols if c in df.columns]}")

# ==========================================
# 🎯 STEP 3: DEFINE TARGET
# ==========================================
target_col = 'Target_2yr'
df = df.dropna(subset=[target_col])
print(f"🎯 Target: {target_col}")
print(f"   Class distribution:\n{df[target_col].value_counts()}")

# ==========================================
# 🧩 STEP 4: SEPARATE FEATURES AND LABELS
# ==========================================
X = df.drop(columns=[target_col])
y = df[target_col]

# ==========================================
# 👩‍🔬 STEP 5: ENCODE CATEGORICAL COLUMNS
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"📝 Encoding categorical columns: {cat_cols}")

for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# ==========================================
# 🧮 STEP 6: ENSURE ALL VALUES ARE NUMERIC
# ==========================================
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

print(f"\n📊 Feature columns ({len(X.columns)}):")
print(X.columns.tolist())

# ==========================================
# 🔍 STEP 7: IDENTIFY CATEGORICAL VS NUMERIC
# ==========================================
# Define which columns should NOT be scaled
# Typically: gender, APOE4 count, and any other categorical encodings
CATEGORICAL_COLS = ['PTGENDER']  # Add other categorical columns as needed

# Check if APOE4_count exists and add it to categorical if it does
if 'APOE4_count' in X.columns:
    CATEGORICAL_COLS.append('APOE4_count')

NUMERIC_COLS = [c for c in X.columns if c not in CATEGORICAL_COLS]

print(f"\n🔢 Numeric columns ({len(NUMERIC_COLS)}): {NUMERIC_COLS}")
print(f"📁 Categorical columns ({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}")

# ==========================================
# 📏 STEP 8: TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✂️ Train-test split:")
print(f"   Training: {X_train.shape[0]} samples")
print(f"   Testing: {X_test.shape[0]} samples")

# ==========================================
# 📈 STEP 9: SCALE ONLY NUMERIC FEATURES
# ==========================================
print(f"\n📈 Scaling numeric features...")

scaler = StandardScaler()

# Extract numeric features
X_train_numeric = X_train[NUMERIC_COLS]
X_test_numeric = X_test[NUMERIC_COLS]

# Fit scaler ONLY on numeric columns
X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
X_test_numeric_scaled = scaler.transform(X_test_numeric)

# Convert back to DataFrame to preserve column names
X_train_numeric_scaled_df = pd.DataFrame(
    X_train_numeric_scaled, 
    columns=NUMERIC_COLS, 
    index=X_train.index
)
X_test_numeric_scaled_df = pd.DataFrame(
    X_test_numeric_scaled, 
    columns=NUMERIC_COLS, 
    index=X_test.index
)

# Add back categorical columns (unscaled)
for col in CATEGORICAL_COLS:
    X_train_numeric_scaled_df[col] = X_train[col].values
    X_test_numeric_scaled_df[col] = X_test[col].values

# Reorder columns to match original order
X_train_final = X_train_numeric_scaled_df[X.columns]
X_test_final = X_test_numeric_scaled_df[X.columns]

print(f"✅ Scaling complete:")
print(f"   X_train shape: {X_train_final.shape}")
print(f"   X_test shape: {X_test_final.shape}")
print(f"   Final column order: {X_train_final.columns.tolist()}")

# ==========================================
# 💾 STEP 10: SAVE SCALER
# ==========================================
joblib.dump(scaler, "models/progress_basic_scaler.pkl")
print(f"\n💾 Scaler saved to: models/progress_basic_scaler.pkl")

# Save feature order for inference
joblib.dump(X.columns.tolist(), "models/progress_basic_features.pkl")
joblib.dump(NUMERIC_COLS, "models/progress_basic_numeric_cols.pkl")
joblib.dump(CATEGORICAL_COLS, "models/progress_basic_categorical_cols.pkl")
print(f"💾 Feature metadata saved")

# ==========================================
# ⚖️ STEP 11: HANDLE CLASS IMBALANCE (SMOTE)
# ==========================================
print(f"\n⚖️ Applying SMOTE to balance classes...")
print(f"   Before SMOTE: {y_train.value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_final, y_train)

print(f"   After SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")
print(f"   Resampled training shape: {X_train_res.shape}")

# ==========================================
# 🤖 STEP 12: DEFINE MODELS
# ==========================================
print(f"\n🤖 Defining models...")

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, 
        class_weight='balanced',
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, 
        random_state=42, 
        class_weight='balanced'
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, 
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42, 
        scale_pos_weight=(y == 0).sum() / (y == 1).sum()
    ),
    "LightGBM": LGBMClassifier(
        random_state=42, 
        class_weight='balanced',
        verbose=-1
    ),
    "SVM": SVC(
        kernel='rbf', 
        probability=True, 
        class_weight='balanced', 
        random_state=42
    )
}

# ==========================================
# 📊 STEP 13: TRAIN AND EVALUATE MODELS
# ==========================================
def get_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive performance metrics"""
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return acc, auc, f1, sensitivity, specificity

print(f"\n🏋️ Training and evaluating models...")
results = []

for name, model in models.items():
    print(f"\n   Training {name}...")
    
    # Train model
    model.fit(X_train_res, y_train_res)
    
    # Predictions
    y_pred = model.predict(X_test_final)
    y_proba = model.predict_proba(X_test_final)[:, 1]
    
    # Calculate metrics
    acc, auc, f1, sens, spec = get_metrics(y_test, y_pred, y_proba)
    
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "ROC-AUC": round(auc, 4),
        "F1-Score": round(f1, 4),
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4)
    })
    
    print(f"      ✓ Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}, F1: {f1:.4f}")

# ==========================================
# 📋 STEP 14: COMPARE MODELS
# ==========================================
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)

print(f"\n{'='*80}")
print(f"📋 MODEL COMPARISON (sorted by ROC-AUC)")
print(f"{'='*80}")
print(results_df.to_string(index=False))
print(f"{'='*80}\n")

# Save results
results_df.to_csv("models/progress_basic_model_comparison.csv", index=False)
print(f"💾 Results saved to: models/progress_basic_model_comparison.csv")

# ==========================================
# 🏆 STEP 15: SAVE BEST MODEL
# ==========================================
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n🏆 Best model: {best_model_name}")
print(f"   ROC-AUC: {results_df.iloc[0]['ROC-AUC']}")
print(f"   Accuracy: {results_df.iloc[0]['Accuracy']}")

# Save model based on type
if best_model_name == "XGBoost":
    best_model.get_booster().save_model("models/progress_basic_XGBoost_best_model.json")
    print(f"💾 Model saved to: models/progress_basic_XGBoost_best_model.json")
else:
    joblib.dump(best_model, f"models/progress_basic_{best_model_name}_best_model.pkl")
    print(f"💾 Model saved to: models/progress_basic_{best_model_name}_best_model.pkl")

# ==========================================
# 🔍 STEP 16: SHAP ANALYSIS (OPTIONAL)
# ==========================================
tree_models = ["XGBoost", "LightGBM", "RandomForest", "GradientBoosting"]

if best_model_name in tree_models:
    print(f"\n🔍 Running SHAP analysis on {best_model_name}...")
    
    try:
        # For XGBoost, reload from saved model
        if best_model_name == "XGBoost":
            import xgboost as xgb
            best_model = xgb.XGBClassifier()
            best_model.load_model("models/progress_basic_XGBoost_best_model.json")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(best_model)
        
        # Use a sample for faster computation
        sample_size = min(1000, len(X_train_res))
        X_sample = X_train_res[:sample_size] if isinstance(X_train_res, np.ndarray) else X_train_res[:sample_size].values
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Create SHAP plots
        print(f"   Generating SHAP plots...")
        
        # Bar plot (feature importance)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, 
                         feature_names=X.columns.tolist())
        plt.tight_layout()
        plt.savefig("models/shap_feature_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   💾 Feature importance plot saved: models/shap_feature_importance.png")
        
        # Summary plot (beeswarm)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False,
                         feature_names=X.columns.tolist())
        plt.tight_layout()
        plt.savefig("models/shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   💾 Summary plot saved: models/shap_summary.png")
        
        print(f"   ✅ SHAP analysis complete")
        
    except Exception as e:
        print(f"   ⚠️ SHAP analysis failed: {e}")

else:
    print(f"\n⚠️ SHAP not applicable: {best_model_name} is not tree-based")
    print(f"   Consider using shap.KernelExplainer for linear models")

# ==========================================
# ✅ TRAINING COMPLETE
# ==========================================
print(f"\n{'='*80}")
print(f"✅ TRAINING PIPELINE COMPLETE")
print(f"{'='*80}")
print(f"📊 Models trained: {len(models)}")
print(f"🏆 Best model: {best_model_name} (ROC-AUC: {results_df.iloc[0]['ROC-AUC']})")
print(f"💾 Artifacts saved to: models/")
print(f"{'='*80}\n")