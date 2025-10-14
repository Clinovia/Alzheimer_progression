import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# Load your dataset
df = pd.read_csv("data/ADNI_core_model_dataset.csv")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 🧹 Step 1. Drop diagnostic & unrelated target columns
drop_cols = ['DIAGNOSIS', 'Target_1yr', 'Target_3yr', 'Target_5yr']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 🎯 Step 2. Define your target
target_col = 'Target_2yr'

# Drop rows with missing target
df = df.dropna(subset=[target_col])

# 🧩 Step 3. Separate features and labels
X = df.drop(columns=[target_col])
y = df[target_col]

# 👩‍🔬 Step 4. Encode categorical columns (e.g., Female/Male)
cat_cols = X.select_dtypes(include=['object']).columns

for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# 🧮 Step 5. Ensure all values are numeric
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)  # replace any remaining NaN with 0

# 📏 Step 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 📈 Step 7. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"✅ Data ready: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=(y == 0).sum() / (y == 1).sum()),
    "LightGBM": LGBMClassifier(random_state=42, class_weight='balanced'),
    "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
}

# Function for performance metrics
def get_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return acc, auc, f1, sensitivity, specificity

# Compare models
results = []
for name, model in models.items():
    if name in ['LogisticRegression', 'SVM']:
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc, auc, f1, sens, spec = get_metrics(y_test, y_pred, y_proba)
    results.append({
        "Model": name,
        "Accuracy": acc,
        "ROC-AUC": auc,
        "F1-Score": f1,
        "Sensitivity (Recall)": sens,
        "Specificity": spec
    })

import joblib

# Summary table
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
print("\nModel Comparison with Sensitivity & Specificity:\n")
print(results_df.to_string(index=False))

#import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import pandas as pd

# -----------------------------
# 1️⃣ Save model comparison results
# -----------------------------
results_df.to_csv("model_comparison_results.csv", index=False)
print("✅ Results saved as model_comparison_results.csv")

# -----------------------------
# 2️⃣ Identify the best model
# -----------------------------
best_model_name = results_df.iloc[0]['Model']
print(f"🏆 Best model: {best_model_name}")
best_model = models[best_model_name]  # already trained in memory

# -----------------------------
# 3️⃣ Save the best model safely
# -----------------------------
if best_model_name == "XGBoost":
    # Use XGBoost's native save/load
    best_model.get_booster().save_model("xgb_best_model.json")
    print("✅ XGBoost model saved as xgb_best_model.json")
elif best_model_name in ["LightGBM", "RandomForest", "GradientBoosting", "LogisticRegression", "SVM"]:
    # Use joblib for non-XGBoost models
    joblib.dump(best_model, f"{best_model_name}_best_model.pkl")
    print(f"✅ {best_model_name} model saved as {best_model_name}_best_model.pkl")
else:
    raise ValueError("Unknown model type for saving.")

# -----------------------------
# 4️⃣ Run SHAP on tree-based models
# -----------------------------
tree_models = ["XGBoost", "LightGBM", "RandomForest", "GradientBoosting"]

if best_model_name in tree_models:
    # For XGBoost, reload from JSON to avoid SHAP Unicode errors
    if best_model_name == "XGBoost":
        best_model = xgb.XGBClassifier()
        best_model.load_model("xgb_best_model.json")

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(tree_models[2])
    shap_values = explainer.shap_values(X_train_res)  # balanced training data after SMOTE

    # Global feature importance (bar plot)
    shap.summary_plot(shap_values, X_train_res, plot_type="bar")

    # Detailed summary plot
    shap.summary_plot(shap_values, X_train_res)

    # Dependence plot for a specific feature
    feature_name = "ADAS13"  # replace with a top feature
    shap.dependence_plot(feature_name, shap_values, X_train_res)

else:
    print(f"⚠️ SHAP not run: {best_model_name} is not tree-based. Use KernelExplainer for other models.")
