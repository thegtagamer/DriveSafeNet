
# 🚗 DriveSafeNet

**DriveSafeNet** is a real-time vision-based deep learning pipeline that detects and classifies road hazards from dashcam footage—including stopped vehicles, bad weather, potholes, and rare obstacles—and overlays predictions with GPS metadata for enhanced road safety and autonomous driving research.

---

## 📌 Features

- 🔍 **Stopped Vehicle Detection** using Faster R-CNN + centroid tracking  
- 🌦 **Weather Classification** using ConvNeXt  
- ⚠️ **Hazard Detection** using YOLOv11 (Lost & Found, RoadObstacle21 datasets)  
- 🕳️ **Pothole Detection** using YOLOv11 (custom dataset)  
- 📍 **Live GPS Overlay** via IP-based geolocation  
- 🎞️ **Real-time Inference Pipeline** with annotated video output

---

## 📁 Directory Structure

```
.
├── realtime_inferv2.py          # Final integrated inference script
├── train_weather.py             # Training script for weather classifier
├── convert_laf_to_yolo_seg.py   # LostAndFound dataset preprocessor
├── roadObs_convert.py           # RoadObstacle21 dataset preprocessor
├── checkpoints/                 # Model weights (.pth and .pt files)
├── data/
│   ├── lost_and_found_yolo/
│   ├── road_obstacle21/
│   └── bdd100k/
├── models/
│   └── weather_classifier.py
```

---

## 🧠 Model Overview

| Module                    | Architecture         | Dataset(s) Used                          |
|---------------------------|----------------------|-------------------------------------------|
| Weather Classifier        | ConvNeXt             | BDD100K (weather labels)                  |
| Stopped Vehicle Detector  | Faster R-CNN         | BDD100K (object detection subset)         |
| Hazard Detector           | YOLOv11              | LostAndFound, RoadObstacle21              |
| Pothole Detector          | YOLOv11              | Custom-labeled pothole dataset            |

---

## 📊 Performance Metrics

| Task                   | Metric                  | Score                     |
|------------------------|--------------------------|----------------------------|
| Weather Detection      | Accuracy                 | ~65%                      |
| Stopped Vehicles       | F1 / mAP@0.5:0.95        | ~71% / 38%                |
| Hazard Detection       | mAP@0.5:0.95             | ~63%                      |
| Pothole Detection      | mAP@0.5:0.95             | ~51%                      |

---

## 🧪 Getting Started

### 1. Install Dependencies

```bash
conda create -n drivesafenet python=3.10
conda activate drivesafenet
pip install torch torchvision opencv-python ultralytics albumentations tqdm
```

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/DriveSafeNet.git
cd DriveSafeNet
```

### 3. Prepare the Data

- BDD100K: https://bdd-data.berkeley.edu
- LostAndFound Dataset: https://www.6d-vision.com/lostandfounddataset
- RoadObstacle21: https://github.com/valeoai/road-obstacle-detection
- Pothole Dataset: https://www.kaggle.com/api/v1/datasets/download/anggadwisunarto/potholes-detection-yolov8

## 4. 🔧 Training

This project includes four dedicated training scripts and two evaluation scripts for different components of DriveSafeNet.

### 4,1 Weather Classification (ConvNeXt)
Train a weather classification model using the BDD100K dataset:

```bash
python train_weather.py \
  --img_dir data/bdd100k/images/100k/train \
  --label_json data/labels/bdd100k/labels/weather_train.json \
  --val_img_dir data/bdd100k/images/100k/val \
  --val_label_json data/labels/bdd100k/labels/weather_val.json \
  --epochs 60 \
  --batch 64 \
  --loss_fn focal \
  --run_name weather-classification
```

Evaluate weather classification:
```bash
python eval_weather.py \
  --img_dir data/bdd100k/images/100k/val \
  --label_json data/labels/bdd100k/labels/weather_val.json \
  --ckpt checkpoints/weather_final_best.pth
```

### 4.2 Stopped Vehicle Detection (Faster R-CNN)
Train a vehicle detection model with Faster R-CNN:
```bash
python train_det.py --config configs/det_config.yaml
```

Evaluate vehicle detection performance:
```bash
python eval_det.py \
  --img_dir data/bdd100k/images/100k/val \
  --label_json data/labels/bdd100k/labels/det_val.json \
  --ckpt checkpoints/stopped_ep9.pth \
  --device cuda \
  --out_dir eval_preds
```

### 4.3 Obstacle Detection (YOLOv11)
Train a YOLOv11 model on RoadObstacle21 data:
```bash
python train_obstacles.py \
  --model yolo11x.pt \
  --data_yaml data/obstacle_data.yaml \
  --epochs 60 \
  --batch 16 \
  --run_name yolov11-road-hazards
```

### 4.4 Pothole Detection (YOLOv11)
Train a YOLOv11 model for pothole detection:
```bash
python train_potholes.py \
  --model yolo11x.pt \
  --data_yaml data/pothole_data.yaml \
  --epochs 150 \
  --batch 16 \
  --run_name yolov11-pothole-detection
```


### 4.5 Run Real-time Inference

```bash
python realtime_inferv2.py   --input_video samples/sample_road.mp4   --det_ckpt checkpoints/stopped_ep9.pth   --weather_ckpt checkpoints/weather_final_best.pth   --hazard_ckpt checkpoints/yolov11-hazards.pt   --pothole_ckpt checkpoints/yolov11-potholes.pt   --device cuda   --out_dir inference_output
```

---

## 📽 Example Output

> Real-time dashcam annotated video with:
> - Weather prediction
> - Tracked vehicle IDs and stopped status
> - Road hazards and potholes
> - Lat/Lon overlay




## 🙋‍♂️ Author

**Abhishek Dey**  
MS CS (ML Concentration) – George Mason University  
[LinkedIn](https://linkedin.com/in/abhishekdey)
