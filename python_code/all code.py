# ============================================================
# COLLEGE ADMISSION CHANCE PREDICTION - MACHINE LEARNING PROJECT
# ============================================================
# Project Members:
# 1. HETVI RAJODIYA - Roll No. 504
# 2. KHUSHI PATEL - Roll No. 505
#
# Dataset: 2500 records | 10 fields
# Target: Admission_Chance (continuous value between 0 and 1)
# Algorithms: Linear Regression, Random Forest, Gradient Boosting
# Best model selected using lowest RMSE on held-out test data.
#
# NOTE: The dataset is synthetic and intended for academic demonstration.
# Predictions are illustrative, not official university admission decisions.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "Dataset", "college_admission_2500_10_fields.csv")
IMAGE_DIR = os.path.join(PROJECT_DIR, "Images of model")
MODEL_DIR = os.path.join(PROJECT_DIR, "Model")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("          COLLEGE ADMISSION CHANCE PREDICTION")
print("=" * 70)
print("HETVI RAJODIYA - Roll No. 504")
print("KHUSHI PATEL - Roll No. 505")

# 1. Load dataset
df = pd.read_csv(DATA_PATH)
print("\\nDataset shape:", df.shape)
print("\\nFirst five records:\\n", df.head())
print("\\nDataset information:")
df.info()
print("\\nMissing values:\\n", df.isnull().sum())
print("\\nDuplicate records:", df.duplicated().sum())

# 2. Basic cleaning
df = df.drop_duplicates().copy()
df = df.dropna()
target = "Admission_Chance"
features = [c for c in df.columns if c != target]
X, y = df[features], df[target]

# 3. Exploratory data analysis
plt.figure(figsize=(7, 4))
plt.hist(y, bins=25)
plt.title("Admission Chance Distribution")
plt.xlabel("Admission Chance")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "admission_chance_distribution.png"))
plt.close()

plt.figure(figsize=(7, 4))
plt.scatter(df["CGPA"], y, alpha=0.35)
plt.title("CGPA vs Admission Chance")
plt.xlabel("CGPA")
plt.ylabel("Admission Chance")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "cgpa_vs_admission_chance.png"))
plt.close()

plt.figure(figsize=(7, 4))
plt.boxplot([df.loc[df.Research == 0, target], df.loc[df.Research == 1, target]],
            labels=["No", "Yes"])
plt.title("Admission Chance by Research Experience")
plt.xlabel("Research Experience")
plt.ylabel("Admission Chance")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "research_vs_admission_chance.png"))
plt.close()

plt.figure(figsize=(7, 4))
plt.hist(df["GRE_Score"], bins=20)
plt.title("GRE Score Distribution")
plt.xlabel("GRE Score")
plt.ylabel("Students")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "gre_score_distribution.png"))
plt.close()

corr = df.corr(numeric_only=True)
plt.figure(figsize=(8, 6))
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=70, ha="right")
plt.yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "correlation_heatmap.png"))
plt.close()

# 4. Train/test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("\\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 5. Train multiple regression models
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=250, learning_rate=0.04, max_depth=2,
        loss="huber", random_state=42
    )
}

results = []
trained_models = {}
for name, model in models.items():
    print("\\nTraining", name)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    row = {
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions)
    }
    results.append(row)
    trained_models[name] = model

results_df = pd.DataFrame(results).sort_values("RMSE")
print("\\nModel comparison:\\n", results_df.to_string(index=False))

# 6. Select best model and evaluate
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\\nBest model:", best_name)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")

# 7. Model comparison graphs
plt.figure(figsize=(8, 4))
x = np.arange(len(results_df))
w = 0.35
plt.bar(x - w/2, results_df["MAE"], w, label="MAE")
plt.bar(x + w/2, results_df["RMSE"], w, label="RMSE")
plt.xticks(x, results_df["Model"])
plt.ylabel("Error")
plt.title("Model Error Comparison (Lower is Better)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "model_error_comparison.png"))
plt.close()

plt.figure(figsize=(8, 4))
plt.bar(results_df["Model"], results_df["R2"])
plt.ylim(0, 1)
plt.ylabel("R-squared")
plt.title("R-squared Comparison (Higher is Better)")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "model_r2_comparison.png"))
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.45)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Actual Admission Chance")
plt.ylabel("Predicted Admission Chance")
plt.title("Actual vs Predicted Admission Chance")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "actual_vs_predicted.png"))
plt.close()

# Feature influence for linear regression (standardized coefficients)
linear_model = trained_models["Linear Regression"]
coefficients = pd.Series(
    np.abs(linear_model.named_steps["regressor"].coef_), index=features
).sort_values()
plt.figure(figsize=(7, 4))
plt.barh(coefficients.index, coefficients.values)
plt.xlabel("Absolute Standardized Coefficient")
plt.title("Feature Influence - Linear Regression")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "feature_influence.png"))
plt.close()

# 8. Save trained model, feature names, and evaluation output
joblib.dump(best_model, os.path.join(MODEL_DIR, "admission_chance_model.joblib"))
joblib.dump(features, os.path.join(MODEL_DIR, "model_features.joblib"))
results_df.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"), index=False)

# 9. Example prediction
sample_student = pd.DataFrame([{
    "GRE_Score": 315, "TOEFL_Score": 105, "University_Rating": 4,
    "SOP_Rating": 4.0, "LOR_Rating": 4.0, "CGPA": 8.5,
    "Research": 1, "Work_Experience": 2, "Internship_Count": 2
}])[features]
sample_prediction = float(best_model.predict(sample_student)[0])
print(f"\\nExample student's predicted admission chance: {sample_prediction:.3f}")
print("\\nProject execution completed successfully.")
