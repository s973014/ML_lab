import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
import joblib

# === 1. Load saved CNN features ===
csv_path = "cnn_features_dataset_svc.csv"  # твой CSV с 2000 фичей
df = pd.read_csv(csv_path)

X = df.drop(columns=["label"]).values
y = df["label"].values

# === 2. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 3. CatBoost Classifier ===
model = CatBoostClassifier(
    iterations=2000,
    depth=6,
    learning_rate=0.02,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=200,
    task_type="CPU"  # если есть GPU; иначе удали эту строку
)

# Train
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# === 4. Evaluation ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# === 5. Save model ===
model.save_model("cnn_ai_detector_catboost.cbm")
print("Model saved: cnn_ai_detector_catboost.cbm")

# === 6. Example prediction ===
example_features = X_test[0].reshape(1, -1)
prob_ai = model.predict_proba(example_features)[0][1]
print(f"\nExample prob AI: {prob_ai:.4f}")
