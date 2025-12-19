import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

# ---------------- Main ----------------
if __name__ == "__main__":
    # Загружаем CSV с признаками
    df = pd.read_csv("cnn_features_dataset_svc.csv")
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # Настройка 5-fold Stratified кросс-валидации
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    acc_list = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        base_svc = LinearSVC(C=1.0, max_iter=5000)
        clf = CalibratedClassifierCV(base_svc, method="sigmoid")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        acc_list.append(acc)

        print(f"\n--- Fold {fold} ---")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred))
        fold += 1

    print(f"\nAverage accuracy across 5 folds: {np.mean(acc_list):.3f}")

    # Обучаем модель на всех данных и сохраняем
    final_base_svc = LinearSVC(C=1.0, max_iter=5000)
    final_clf = CalibratedClassifierCV(final_base_svc, method="sigmoid")
    final_clf.fit(X, y)
    joblib.dump(final_clf, "cnn_ai_detector_svc.pkl")
    print("Final LinearSVC model trained on full dataset and saved: cnn_ai_detector_svc.pkl")
