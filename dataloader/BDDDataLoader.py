import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

# Hazard label mapping for detection
hazard_label_map = {
    'car': 0,
    'bus': 1,
    'truck': 2,
    'train': 3,
    'bike': 4,
    'motor': 5,
    'rider': 6,
    'person': 7,
    'traffic light': 8,
    'traffic sign': 9
}

# Weather label mapping
weather_map = {
    'clear': 0,
    'partly cloudy': 1,
    'overcast': 2,
    'rainy': 3,
    'snowy': 4,
    'foggy': 5
}


class BDD100KDataset(Dataset):
    def __init__(self, image_dir, label_json_path, transform=None, filter_weather=None):
        """
        :param image_dir: Path to directory with .jpg images
        :param label_json_path: Path to .json file containing image-level annotations
        :param transform: Albumentations or torchvision transforms
        :param filter_weather: Optional set of weather types to filter (e.g. {'foggy', 'snowy'})
        """
        self.image_dir = image_dir
        self.transform = transform

        with open(label_json_path, 'r') as f:
            all_annotations = json.load(f)

        if filter_weather:
            self.annotations = [
                ann for ann in all_annotations
                if ann.get("attributes", {}).get("weather") in filter_weather
            ]
        else:
            self.annotations = all_annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_name = ann["name"]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []

        for obj in ann.get("labels", []):
            category = obj["category"]
            if category in hazard_label_map and "box2d" in obj:
                box = obj["box2d"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                boxes.append([x1, y1, x2, y2])
                labels.append(hazard_label_map[category])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        weather_str = ann.get("attributes", {}).get("weather", "clear")
        weather_label = weather_map.get(weather_str, 0)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "weather": torch.tensor([weather_label])
        }

        if self.transform:
            image = self.transform(image)

        return image, target
