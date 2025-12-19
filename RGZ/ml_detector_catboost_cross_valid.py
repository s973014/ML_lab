import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from catboost import CatBoostClassifier
import joblib

# === 1. Load saved CNN features ===
csv_path = "cnn_features_dataset_svc.csv"  # твой CSV с 2000 фичей
df = pd.read_csv(csv_path)
X = df.drop(columns=["label"]).values
y = df["label"].values

# === 2. 5-fold Stratified Cross-Validation ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_list, precision_list, recall_list = [], [], []
fold = 1

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = CatBoostClassifier(
        iterations=2000,
        depth=6,
        learning_rate=0.02,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=200,
        task_type="CPU"  # если есть GPU, можно поставить "GPU"
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    accuracy_list.append(acc)
    precision_list.append(prec)
    recall_list.append(rec)

    print(f"\n--- Fold {fold} ---")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(classification_report(y_test, y_pred))
    fold += 1

# --- Средние метрики по 5 фолдам ---
print("\n=== Average metrics across 5 folds ===")
print(f"Average Accuracy: {np.mean(accuracy_list):.3f}")
print(f"Average Precision: {np.mean(precision_list):.3f}")
print(f"Average Recall: {np.mean(recall_list):.3f}")
