from dataloader.BDDDataLoader import BDD100KDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Reverse label maps for display
inv_hazard_map = {
    0: 'car', 1: 'bus', 2: 'truck', 3: 'train',
    4: 'bike', 5: 'motor', 6: 'rider', 7: 'person',
    8: 'traffic light', 9: 'traffic sign'
}

inv_weather_map = {
    0: 'clear', 1: 'partly cloudy', 2: 'overcast',
    3: 'rainy', 4: 'snowy', 5: 'foggy'
}

# 🔁 Albumentations transforms
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

# ✅ Load dataset
dataset = BDD100KDataset(
    image_dir="./data/bdd10k/train",
    label_json_path="./data/bdd10k/labels/bdd100k_labels_images_train.json",
    transform=transform
)

# 🔄 PyTorch Dataloader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# 🔎 Test 1 sample
images, targets = next(iter(dataloader))
image = images[0].permute(1, 2, 0).cpu().numpy()  # [C,H,W] -> [H,W,C]
image = (image * 255).astype(np.uint8)

boxes = targets[0]["boxes"]
labels = targets[0]["labels"]
weather = inv_weather_map[targets[0]["weather"].item()]

# Draw boxes
for box, label in zip(boxes, labels):
    x1, y1, x2, y2 = map(int, box.tolist())
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, inv_hazard_map[label.item()], (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

# Show image
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.title(f"Sample with Hazards | Weather: {weather}")
plt.axis("off")
plt.show()