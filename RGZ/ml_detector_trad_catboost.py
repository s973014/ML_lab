import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# ---------------- Load traditional features ----------------
csv_path = "traditional_features_dataset_rnd_fr.csv"
df = pd.read_csv(csv_path)

X = df.drop(columns=["label"]).values
y = df["label"].values

# ---------------- 5-Fold Stratified Cross-Validation ----------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_list, precision_list, recall_list = [], [], []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = CatBoostClassifier(
        iterations=2000,
        depth=6,
        learning_rate=0.02,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=0,       # чтобы не засорять вывод
        task_type="CPU"  # если есть GPU, можно заменить на "GPU"
    )
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)

    y_pred = clf.predict(X_test)
    
    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred))

print("\n--- Cross-Validation Metrics (5 folds) ---")
print(f"Accuracy: {np.mean(accuracy_list):.3f}")
print(f"Precision: {np.mean(precision_list):.3f}")
print(f"Recall: {np.mean(recall_list):.3f}")

# ---------------- Train final model on all data ----------------
final_clf = CatBoostClassifier(
    iterations=2000,
    depth=6,
    learning_rate=0.02,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=200,
    task_type="CPU"
)
final_clf.fit(X, y)
final_clf.save_model("traditional_catboost_model.cbm")
print("Final CatBoost model trained and saved.")
