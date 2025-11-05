"""
train_extended_model.py
ADNI Extended Model Training with Missing Value Handling + SHAP
Properly handles numeric vs categorical feature scaling
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# ==========================================
# 📂 STEP 1: LOAD DATASET
# ==========================================
print("="*80)
print("📂 LOADING EXTENDED DATASET")
print("="*80)

df = pd.read_csv("data/ADNI_extended_model_dataset.csv")
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

# ==========================================
# 🧹 STEP 2: DROP UNNECESSARY COLUMNS
# ==========================================
drop_cols = [
    "RID", "VISCODE_x", "VISCODE_y", "VISCODE", 
    "EXAMDATE", "DIAGNOSIS", "AD_binary", 
    "Target_1yr", "Target_3yr", "Target_5yr"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
print(f"🗑️  Dropped columns: {[c for c in drop_cols if c in df.columns]}")

# ==========================================
# 🎯 STEP 3: DEFINE TARGET
# ==========================================
target_col = 'Target_2yr'
initial_len = len(df)
df = df.dropna(subset=[target_col])
print(f"\n🎯 Target: {target_col}")
print(f"   Rows after dropping missing target: {len(df)} (dropped {initial_len - len(df)})")
print(f"   Class distribution:\n{df[target_col].value_counts()}")

# ==========================================
# 🧩 STEP 4: SEPARATE FEATURES AND TARGET
# ==========================================
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\n📊 Initial features: {len(X.columns)} columns")

# ==========================================
# 👩‍🔬 STEP 5: ENCODE CATEGORICAL VARIABLES
# ==========================================
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"\n📝 Encoding {len(cat_cols)} categorical columns: {cat_cols}")

for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# ==========================================
# 🧮 STEP 6: HANDLE MISSING VALUES
# ==========================================
print(f"\n🔍 HANDLING MISSING VALUES")
print(f"   Initial shape: {X.shape}")

# Check missing value percentage per column
missing_pct = X.isna().mean() * 100
cols_with_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)

if len(cols_with_missing) > 0:
    print(f"\n   Columns with missing values:")
    for col, pct in cols_with_missing.head(10).items():
        print(f"      {col}: {pct:.1f}%")

# Drop columns with >50% missing values
high_missing_cols = missing_pct[missing_pct >= 50].index.tolist()
if high_missing_cols:
    print(f"\n   ❌ Dropping {len(high_missing_cols)} columns with ≥50% missing values:")
    print(f"      {high_missing_cols}")
    X = X.drop(columns=high_missing_cols)

# Fill remaining NaNs with median
remaining_missing = X.isna().sum().sum()
if remaining_missing > 0:
    print(f"\n   🔧 Filling {remaining_missing} remaining NaN values with column medians")
    X = X.fillna(X.median())

# Ensure y matches X after cleaning
y = y.loc[X.index]

print(f"   ✅ Final shape after cleaning: {X.shape}")
print(f"\n📋 FINAL FEATURE COLUMNS ({len(X.columns)}):")
print("   " + "\n   ".join(X.columns.tolist()))

# ==========================================
# 🔍 STEP 7: IDENTIFY CATEGORICAL VS NUMERIC
# ==========================================
# After encoding, identify which columns should NOT be scaled
# Typically: gender, APOE4, race, and other inherently categorical variables

# Common categorical columns in ADNI (adjust based on your data)
CATEGORICAL_COLS = []

# Check for common categorical column names
possible_categorical = ['PTGENDER', 'APOE4', 'APOE4_count', 'PTRACCAT', 'PTETHCAT', 'PTMARRY']
for col in possible_categorical:
    if col in X.columns:
        CATEGORICAL_COLS.append(col)

# Identify columns with very few unique values (likely categorical)
for col in X.columns:
    if col not in CATEGORICAL_COLS:
        n_unique = X[col].nunique()
        if n_unique <= 5 and n_unique > 1:  # Likely categorical with few categories
            print(f"   ℹ️  Detected potential categorical column: {col} ({n_unique} unique values)")
            CATEGORICAL_COLS.append(col)

NUMERIC_COLS = [c for c in X.columns if c not in CATEGORICAL_COLS]

print(f"\n🔢 Feature categorization:")
print(f"   Numeric columns ({len(NUMERIC_COLS)}): {NUMERIC_COLS[:10]}{'...' if len(NUMERIC_COLS) > 10 else ''}")
print(f"   Categorical columns ({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}")

# ==========================================
# 📏 STEP 8: TRAIN-TEST SPLIT
# ==========================================
print(f"\n{'='*80}")
print(f"✂️  TRAIN-TEST SPLIT")
print(f"{'='*80}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")
print(f"   Training class distribution: {dict(y_train.value_counts())}")
print(f"   Test class distribution: {dict(y_test.value_counts())}")

# ==========================================
# 📈 STEP 9: SCALE ONLY NUMERIC FEATURES
# ==========================================
print(f"\n{'='*80}")
print(f"📈 FEATURE SCALING")
print(f"{'='*80}")

scaler = StandardScaler()

if len(NUMERIC_COLS) > 0:
    # Extract numeric features
    X_train_numeric = X_train[NUMERIC_COLS]
    X_test_numeric = X_test[NUMERIC_COLS]
    
    # Fit scaler ONLY on numeric columns
    X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
    X_test_numeric_scaled = scaler.transform(X_test_numeric)
    
    # Convert back to DataFrame
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
    
    print(f"   ✅ Scaled {len(NUMERIC_COLS)} numeric features")
    print(f"   ✅ Preserved {len(CATEGORICAL_COLS)} categorical features")
else:
    # No numeric columns to scale
    X_train_final = X_train
    X_test_final = X_test
    print(f"   ℹ️  No numeric columns to scale")

print(f"   Final training shape: {X_train_final.shape}")
print(f"   Final test shape: {X_test_final.shape}")

# ==========================================
# 💾 STEP 10: SAVE PREPROCESSING ARTIFACTS
# ==========================================
joblib.dump(scaler, "models/progress_advanced_scaler.pkl")
joblib.dump(X.columns.tolist(), "models/progress_advanced_features.pkl")
joblib.dump(NUMERIC_COLS, "models/progress_advanced_numeric_cols.pkl")
joblib.dump(CATEGORICAL_COLS, "models/progress_advanced_categorical_cols.pkl")

print(f"\n💾 SAVED PREPROCESSING ARTIFACTS:")
print(f"   ✓ models/progress_advanced_scaler.pkl")
print(f"   ✓ models/progress_advanced_features.pkl")
print(f"   ✓ models/progress_advanced_numeric_cols.pkl")
print(f"   ✓ models/progress_advanced_categorical_cols.pkl")

# ==========================================
# ⚖️ STEP 11: HANDLE CLASS IMBALANCE (SMOTE)
# ==========================================
print(f"\n{'='*80}")
print(f"⚖️  APPLYING SMOTE")
print(f"{'='*80}")

print(f"   Before SMOTE: {dict(y_train.value_counts())}")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_final, y_train)

print(f"   After SMOTE: {dict(pd.Series(y_train_res).value_counts())}")
print(f"   Resampled training shape: {X_train_res.shape}")

# ==========================================
# 🤖 STEP 12: DEFINE MODELS
# ==========================================
print(f"\n{'='*80}")
print(f"🤖 DEFINING MODELS")
print(f"{'='*80}")

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

print(f"   Defined {len(models)} models: {list(models.keys())}")

# ==========================================
# 📊 STEP 13: EVALUATION FUNCTION
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

# ==========================================
# 🏋️ STEP 14: TRAIN AND EVALUATE
# ==========================================
print(f"\n{'='*80}")
print(f"🏋️  TRAINING AND EVALUATING MODELS")
print(f"{'='*80}")

results = {}
metrics_list = []

for name, model in models.items():
    print(f"\n   Training {name}...")
    
    # Train
    model.fit(X_train_res, y_train_res)
    
    # Predict
    y_pred = model.predict(X_test_final)
    y_proba = model.predict_proba(X_test_final)[:, 1]
    
    # Evaluate
    acc, auc, f1, sens, spec = get_metrics(y_test, y_pred, y_proba)
    
    results[name] = model
    metrics_list.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "ROC-AUC": round(auc, 4),
        "F1-Score": round(f1, 4),
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4)
    })
    
    print(f"      ✓ Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}, F1: {f1:.4f}")

# ==========================================
# 📋 STEP 15: COMPARE MODELS
# ==========================================
results_df = pd.DataFrame(metrics_list).sort_values(by="ROC-AUC", ascending=False)

print(f"\n{'='*80}")
print(f"📊 MODEL COMPARISON RESULTS (sorted by ROC-AUC)")
print(f"{'='*80}")
print(results_df.to_string(index=False))
print(f"{'='*80}")

# Save results
results_df.to_csv("models/progress_advanced_model_comparison.csv", index=False)
print(f"\n💾 Results saved to: models/progress_advanced_model_comparison.csv")

# ==========================================
# 🏆 STEP 16: SAVE BEST MODEL
# ==========================================
best_model_name = results_df.iloc[0]['Model']
best_model = results[best_model_name]

print(f"\n{'='*80}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"{'='*80}")
print(f"   ROC-AUC: {results_df.iloc[0]['ROC-AUC']}")
print(f"   Accuracy: {results_df.iloc[0]['Accuracy']}")
print(f"   F1-Score: {results_df.iloc[0]['F1-Score']}")
print(f"   Sensitivity: {results_df.iloc[0]['Sensitivity']}")
print(f"   Specificity: {results_df.iloc[0]['Specificity']}")

# Save model
if best_model_name == "XGBoost":
    best_model.get_booster().save_model(f"models/progress_advanced_{best_model_name}_best_model.json")
    print(f"\n💾 Model saved to: models/progress_advanced_{best_model_name}_best_model.json")
else:
    joblib.dump(best_model, f"models/progress_advanced_{best_model_name}_best_model.pkl")
    print(f"\n💾 Model saved to: models/progress_advanced_{best_model_name}_best_model.pkl")

# ==========================================
# 🔍 STEP 17: SHAP EXPLAINABILITY
# ==========================================
tree_models = ["XGBoost", "RandomForest", "LightGBM", "GradientBoosting"]

if best_model_name in tree_models:
    print(f"\n{'='*80}")
    print(f"🔍 GENERATING SHAP FEATURE IMPORTANCE")
    print(f"{'='*80}")
    
    try:
        # For XGBoost, reload from saved model
        if best_model_name == "XGBoost":
            import xgboost as xgb
            best_model = xgb.XGBClassifier()
            best_model.load_model(f"models/progress_advanced_{best_model_name}_best_model.json")
        
        # Use a sample for faster computation
        sample_size = min(1000, len(X_train_res))
        X_sample = X_train_res[:sample_size] if isinstance(X_train_res, np.ndarray) else X_train_res[:sample_size]
        
        print(f"   Computing SHAP values on {sample_size} samples...")
        
        # Create TreeExplainer (more stable for tree models)
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification (shap_values might be a list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Feature importance bar plot
        print(f"   Generating feature importance plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False,
                         feature_names=X.columns.tolist())
        plt.tight_layout()
        plt.savefig("models/shap_advanced_feature_importance_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Saved: models/shap_advanced_feature_importance_bar.png")
        
        # Detailed summary plot
        print(f"   Generating summary plot...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, show=False,
                         feature_names=X.columns.tolist())
        plt.tight_layout()
        plt.savefig("models/shap_advanced_summary_plot.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Saved: models/shap_advanced_summary_plot.png")
        
        print(f"\n   ✅ SHAP analysis complete")
        
    except Exception as e:
        print(f"\n   ⚠️ SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()

else:
    print(f"\n⚠️ SHAP not applicable for {best_model_name}")
    print(f"   Consider using shap.KernelExplainer for non-tree models")

# ==========================================
# ✅ TRAINING COMPLETE
# ==========================================
print(f"\n{'='*80}")
print(f"✅ EXTENDED MODEL TRAINING COMPLETE")
print(f"{'='*80}")
print(f"📊 Models trained: {len(models)}")
print(f"🏆 Best model: {best_model_name} (ROC-AUC: {results_df.iloc[0]['ROC-AUC']})")
print(f"📁 Final features: {len(X.columns)} columns")
print(f"💾 All artifacts saved to: models/")
print(f"{'='*80}\n")