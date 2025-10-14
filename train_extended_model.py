"""
train_extended_model.py
ADNI Extended Model Training with Missing Value Handling + SHAP
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

# -----------------------------------------------------
# 1️⃣ Load dataset
# -----------------------------------------------------
df = pd.read_csv("data/ADNI_extended_model_dataset.csv")

# Drop unnecessary or diagnostic columns
drop_cols = ['DIAGNOSIS', 'Target_1yr', 'Target_3yr', 'Target_5yr']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Target column
target_col = 'Target_2yr'
df = df.dropna(subset=[target_col])

# -----------------------------------------------------
# 2️⃣ Split features and target
# -----------------------------------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# -----------------------------------------------------
# 3️⃣ Encode categorical variables
# -----------------------------------------------------
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# -----------------------------------------------------
# 4️⃣ Handle Missing Values
# -----------------------------------------------------
# Drop columns with too many NaNs (e.g., >50%)
X = X.loc[:, X.isna().mean() < 0.5]

# Fill remaining NaNs with median
X = X.fillna(X.median())

# Ensure y matches X after cleaning
y = y.loc[X.index]

# -----------------------------------------------------
# 5️⃣ Train/Test Split + Scaling + SMOTE
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# -----------------------------------------------------
# 6️⃣ Define Models
# -----------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, eval_metric='logloss', random_state=42,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum()
    ),
    "LightGBM": LGBMClassifier(random_state=42, class_weight='balanced'),
    "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
}

# -----------------------------------------------------
# 7️⃣ Evaluation Function
# -----------------------------------------------------
def get_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return acc, auc, f1, sensitivity, specificity

# -----------------------------------------------------
# 8️⃣ Train + Evaluate
# -----------------------------------------------------
results = {}
metrics_list = []

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc, auc, f1, sens, spec = get_metrics(y_test, y_pred, y_proba)
    results[name] = model
    metrics_list.append({
        "Model": name,
        "Accuracy": acc,
        "ROC-AUC": auc,
        "F1-Score": f1,
        "Sensitivity (Recall)": sens,
        "Specificity": spec
    })

results_df = pd.DataFrame(metrics_list).sort_values(by="ROC-AUC", ascending=False)
print("\n📊 Model Comparison Results:\n")
print(results_df.to_string(index=False))

# Save performance results
results_df.to_csv("extended_model_comparison_results.csv", index=False)
print("\n✅ Results saved as extended_model_comparison_results.csv")

# -----------------------------------------------------
# 9️⃣ Save Best Model
# -----------------------------------------------------
best_model_name = results_df.iloc[0]['Model']
best_model = results[best_model_name]
print(f"\n🏆 Best Model: {best_model_name}")

joblib.dump(best_model, f"{best_model_name}_best_model.pkl")
print(f"✅ Saved as {best_model_name}_best_model.pkl")

# -----------------------------------------------------
# 🔟 SHAP Explainability (Tree-based models only)
# -----------------------------------------------------
if best_model_name in ["XGBoost", "RandomForest", "LightGBM", "GradientBoosting"]:
    print("\n🔍 Generating SHAP feature importance...")
    explainer = shap.Explainer(best_model, X_train_res)
    shap_values = explainer(X_train_res)

    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig("shap_feature_importance_bar.png", bbox_inches="tight")
    print("📈 Saved SHAP feature importance plot as shap_feature_importance_bar.png")

    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("shap_summary_plot.png", bbox_inches="tight")
    print("📊 Saved SHAP summary plot as shap_summary_plot.png")

print("\n🎯 Training + Explainability Complete.")
