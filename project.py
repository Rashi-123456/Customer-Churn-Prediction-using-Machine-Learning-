import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample

# 1. Data Loading & Cleaning

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.drop(columns=["customerID"], inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

# 2. Train-Test Split (Stratified)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Downsampling (Handling Class Imbalance)

train_df = X_train.copy()
train_df["Churn"] = y_train

majority = train_df[train_df.Churn == 0]
minority = train_df[train_df.Churn == 1]

majority_downsampled = resample(
    majority,
    replace=False,
    n_samples=len(minority),
    random_state=42
)

downsampled = pd.concat([majority_downsampled, minority])

X_train_down = downsampled.drop("Churn", axis=1)
y_train_down = downsampled["Churn"]

# 4. Preprocessing Pipeline

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ]
)

# 5. Model Selection using Stratified K-Fold

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

print("\nModel Selection (Stratified K-Fold ROC-AUC)\n")

best_model_name = ""
best_score = 0

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    scores = cross_val_score(pipeline, X_train_down, y_train_down, cv=skf, scoring="roc_auc")
    mean_score = scores.mean()

    print(f"{name}: ROC-AUC = {mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name

print(f"\nBest Model Selected: {best_model_name}")

# 6. Hyperparameter Tuning (Gradient Boosting)

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__max_depth": [3, 4, 5]
}

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(random_state=42))
])

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=skf,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_down, y_train_down)

print("\nBest Hyperparameters:")
print(random_search.best_params_)

final_model = random_search.best_estimator_

# 7. Overfitting Check

train_pred = final_model.predict(X_train_down)
test_pred = final_model.predict(X_test)

train_accuracy = accuracy_score(y_train_down, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("\nOverfitting Analysis")
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# 8. Final Evaluation

y_prob = final_model.predict_proba(X_test)[:, 1]

print("\nFinal Model Performance")
print("Test Accuracy:", test_accuracy)
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, test_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, test_pred))

# 9. ROC Curve

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Customer Churn")
plt.show()

# 10. Feature Importance

classifier = final_model.named_steps["classifier"]
importances = classifier.feature_importances_

feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

print("\nTop 10 Important Features:")
print(feature_importance)
