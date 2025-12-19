import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from scipy.stats import kurtosis, entropy

# ---------------- Feature Extractor ----------------
class TraditionalFeatureExtractor:
    def extract_features(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (512, 512))

        features = {}
        dct_blocks = self.extract_dct(gray)

        features["zero_rate"] = self.zero_rate(dct_blocks)
        features["kurtosis"] = self.kurtosis_score(dct_blocks)
        features["band_entropy"] = self.band_entropy(dct_blocks)
        features["high_freq_std"] = self.high_freq_std(dct_blocks)
        features["local_noise"] = self.local_noise(gray)
        features["gradient_std"] = self.gradient_std(gray)
        features["texture_anomaly"] = self.texture_anomaly(gray)

        return np.array([features[k] for k in sorted(features.keys())]), features

    def extract_dct(self, gray):
        h, w = gray.shape
        h -= h % 8
        w -= w % 8
        gray = gray[:h, :w]
        blocks = []
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = np.float32(gray[i:i+8, j:j+8]) - 128
                blocks.append(cv2.dct(block))
        return np.array(blocks)

    def zero_rate(self, dct_blocks):
        flat = dct_blocks.reshape(-1)
        zeros = np.sum(np.abs(flat) < 0.5)
        return zeros / len(flat)

    def kurtosis_score(self, dct_blocks):
        flat = dct_blocks.reshape(-1)
        return np.tanh(np.log1p(abs(kurtosis(flat))) / 5)

    def band_entropy(self, dct_blocks):
        low = dct_blocks[:, :4, :4].reshape(-1)
        high = dct_blocks[:, 4:, 4:].reshape(-1)
        h_low, _ = np.histogram(low, bins=100, range=(-50,50), density=True)
        h_high, _ = np.histogram(high, bins=100, range=(-50,50), density=True)
        return entropy(h_high+1e-12) / (entropy(h_low+1e-12) + 1e-12)

    def high_freq_std(self, dct_blocks):
        high = dct_blocks[:, 4:, 4:].reshape(len(dct_blocks), -1)
        var_per_block = np.var(high, axis=1)
        return np.std(var_per_block)

    def local_noise(self, gray):
        h, w = gray.shape
        h -= h % 16
        w -= w % 16
        gray = gray[:h, :w]
        blocks = []
        for i in range(0, h, 16):
            for j in range(0, w, 16):
                blocks.append(np.std(gray[i:i+16, j:j+16]))
        return np.mean(blocks)

    def gradient_std(self, gray):
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        return np.std(grad_mag)

    def texture_anomaly(self, gray):
        block_size = 16
        h, w = gray.shape
        h -= h % block_size
        w -= w % block_size
        gray = gray[:h, :w]
        anomalies = []
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                hist, _ = np.histogram(block, bins=16, range=(0,255), density=True)
                anomalies.append(entropy(hist + 1e-12))
        anomalies = np.array(anomalies)
        return np.std(anomalies) / (np.mean(anomalies)+1e-12)


# ---------------- Collect features ----------------
def collect_features(folder, label):
    extractor = TraditionalFeatureExtractor()
    feature_list = []
    file_paths = glob(os.path.join(folder, "*.*"))
    for img_path in file_paths:
        try:
            fv, _ = extractor.extract_features(img_path)
            feature_list.append(list(fv) + [label])
        except Exception as e:
            print(f"Error {img_path}: {e}")
    return feature_list


# ---------------- Main ----------------
if __name__ == "__main__":
    real_folder = "dataset/RealArt/RealArt"
    ai_folder = "dataset/AiArtData"

    print("Collecting features for real images...")
    real_features = collect_features(real_folder, label=0)
    print("Collecting features for AI images...")
    ai_features = collect_features(ai_folder, label=1)

    columns = sorted([
        "band_entropy",
        "gradient_std",
        "high_freq_std",
        "kurtosis",
        "local_noise",
        "texture_anomaly",
        "zero_rate"
    ]) + ["label"]

    df = pd.DataFrame(real_features + ai_features, columns=columns)
    df.to_csv("traditional_features_dataset_rnd_fr.csv", index=False)

    # ---------------- 5-Fold Cross-Validation ----------------
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracy_list, precision_list, recall_list = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy_list.append(accuracy_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred))
        recall_list.append(recall_score(y_test, y_pred))

    print("\n--- Cross-Validation Metrics (5 folds) ---")
    print(f"Accuracy: {np.mean(accuracy_list):.3f}")
    print(f"Precision: {np.mean(precision_list):.3f}")
    print(f"Recall: {np.mean(recall_list):.3f}")

    # ---------------- Train final model on all data ----------------
    final_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    final_clf.fit(X, y)
    joblib.dump(final_clf, "traditional_rf_model.pkl")
    print("Final RandomForest model trained and saved.")
