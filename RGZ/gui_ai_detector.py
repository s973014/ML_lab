import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import joblib

# CatBoost
from catboost import CatBoostClassifier

# Torch + ResNet50
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image as PIL_Image


# ---------------- CNN Feature Extractor ----------------
class CNNFeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()  # убираем последний слой
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, img_path):
        img = PIL_Image.open(img_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_t)
        return features.cpu().numpy().flatten()


# ---------------- Tkinter GUI ----------------
class AIDetectorGUI:
    def __init__(self, root):
        self.root = root
        root.title("AI Image Detector")
        root.geometry("550x600")

        self.extractor = CNNFeatureExtractor()

        # --- выбор классификатора ---
        tk.Label(root, text="Выберите классификатор:", font=("Arial", 12)).pack(pady=5)

        self.clf_var = tk.StringVar(value="rf")
        tk.Radiobutton(root, text="Random Forest", variable=self.clf_var, value="rf").pack()
        tk.Radiobutton(root, text="CatBoost", variable=self.clf_var, value="catboost").pack()

        # --- кнопка выбора изображения ---
        tk.Button(root, text="Выбрать изображение", command=self.load_image).pack(pady=10)

        # --- место под изображение ---
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        # --- кнопка проверки ---
        tk.Button(root, text="Проверить изображение", command=self.predict_image,
                  font=("Arial", 12)).pack(pady=10)

        # --- вывод результата ---
        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.pack(pady=15)

        self.image_path = None
        self.tk_image = None

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return

        self.image_path = path

        # показать превью
        img = Image.open(path)
        img.thumbnail((350, 350))
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_image)

    def predict_image(self):
        if not self.image_path:
            messagebox.showerror("Ошибка", "Сначала выберите изображение.")
            return

        try:
            # --- извлечение признаков ---
            features = self.extractor.extract_features(self.image_path).reshape(1, -1)

            # --- загрузка модели ---
            clf_type = self.clf_var.get()

            if clf_type == "rf":
                model_path = "cnn_ai_detector_rf.pkl"
                if not os.path.exists(model_path):
                    raise FileNotFoundError("Файл модели RF не найден!")
                model = joblib.load(model_path)

            elif clf_type == "catboost":
                model_path = "cnn_ai_detector_catboost.cbm"
                if not os.path.exists(model_path):
                    raise FileNotFoundError("Файл модели CatBoost не найден!")
                model = CatBoostClassifier()
                model.load_model(model_path)

            else:
                raise ValueError("Неизвестный классификатор!")

            # --- вероятность ---
            prob_ai = float(model.predict_proba(features)[0][1])

            self.result_label.config(
                text=f"Вероятность ИИ: {prob_ai * 100:.2f}%",
                fg=("red" if prob_ai > 0.5 else "green")
            )

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


# ---------------- Main ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AIDetectorGUI(root)
    root.mainloop()
