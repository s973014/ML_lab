import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------- CNN Feature Extractor ----------------
class CNNFeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Предобученная ResNet50 без классификатора
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()  # убираем последний fully-connected слой
        self.model.eval()
        self.model.to(self.device)

        # Преобразования изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_t)
        return features.cpu().numpy().flatten()


# ---------------- Collect features ----------------
def collect_cnn_features(image_folder, label):
    extractor = CNNFeatureExtractor()
    feature_list = []
    file_paths = glob(os.path.join(image_folder, "*.*"))
    for img_path in file_paths:
        try:
            features = extractor.extract_features(img_path)
            feature_list.append(list(features) + [label])
            print(f"Processed: {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return feature_list


# ---------------- Main ----------------
if __name__ == "__main__":
    # Папки с изображениями
    real_folder = "dataset/RealArt/RealArt"
    ai_folder = "dataset/AiArtData"

    # Сбор признаков
    print("Collecting real images features...")
    real_features = collect_cnn_features(real_folder, label=0)
    print("Collecting AI images features...")
    ai_features = collect_cnn_features(ai_folder, label=1)

    # Создание DataFrame
    feature_len = len(real_features[0]) - 1
    columns = [f"feat_{i}" for i in range(feature_len)] + ["label"]
    df = pd.DataFrame(real_features + ai_features, columns=columns)
    df.to_csv("cnn_features_dataset.csv", index=False)
    print("Features CSV saved: cnn_features_dataset.csv")

    # ---------------- Train/Test Split ----------------
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)

    # ---------------- Train Classifier ----------------
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # ---------------- Test ----------------
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]

    print("\n--- Classification Report on Test Set ---")
    print(classification_report(y_test, y_pred))

    # ---------------- Save model ----------------
    joblib.dump(clf, "cnn_ai_detector.pkl")
    print("Model saved: cnn_ai_detector.pkl")

    # ---------------- Example Prediction ----------------
    test_image = os.path.join(ai_folder, os.listdir(ai_folder)[0])
    extractor = CNNFeatureExtractor()
    feature_vector = extractor.extract_features(test_image).reshape(1, -1)
    prob_ai = clf.predict_proba(feature_vector)[0][1]
    print(f"\nTest image: {test_image}")
    print(f"Probability AI: {prob_ai:.3f}")
