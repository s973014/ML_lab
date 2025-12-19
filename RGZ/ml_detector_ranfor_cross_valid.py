import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import joblib
import numpy as np

# ---------------- Main ----------------
if __name__ == "__main__":
    # Загружаем CSV с признаками
    df = pd.read_csv("cnn_features_dataset.csv")
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # Настройка 5-fold кросс-валидации
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    accuracy_list = []
    precision_list = []
    recall_list = []


    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        accuracy_list.append(acc)
        precision_list.append(prec)
        recall_list.append(rec)

        print(f"\n--- Fold {fold} ---")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred))
        fold += 1

    print("\n=== Average metrics across 5 folds ===")
    print(f"Average Accuracy: {np.mean(accuracy_list):.3f}")
    print(f"Average Precision: {np.mean(precision_list):.3f}")
    print(f"Average Recall: {np.mean(recall_list):.3f}")

    # Обучаем модель на всех данных и сохраняем
    final_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    final_clf.fit(X, y)
    joblib.dump(final_clf, "cnn_ai_detector_rf.pkl")
    print("Final model trained on full dataset and saved: cnn_ai_detector_rf.pkl")
