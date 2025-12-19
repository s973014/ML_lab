import os
import csv
from glob import glob
from ml_detector_randomforest import MLFeatureExtractor  # импортируем из предыдущего кода

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
        img = Image.open(img_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_t)
        return features.cpu().numpy().flatten()

if __name__ == "__main__":
    # Пример использования:
    # Папка с реальными фото
    collect_features("dataset/AiArtData", label=0, output_csv="features_dataset.csv")
    # Папка с AI-сгенерированными фото
    collect_features("dataset/RealArt/RealArt", label=1, output_csv="features_dataset.csv")
