📄 README.md — Full Content
markdown
Copy
# 🚘 Object Detection and Annotation for Autonomous Driving using YOLOv8

This project is part of my Master's program at the University of Limerick. It focuses on developing a **real-time object detection system** for autonomous vehicle safety using the **YOLOv8 model** trained on the **NuScenes dataset**.

---

## 📌 Project Overview

🎯 **Goal**: Build a scalable, real-time object detection system capable of identifying critical road elements such as:
- Cars
- Pedestrians
- Trucks
- Traffic cones
- Buses, and more

⚙️ **Architecture**:
- YOLOv8 (`ultralytics`) as the core detection model
- Trained on the NuScenes mini dataset (`CAM_FRONT` camera view)
- Designed for deployment on hardware like NVIDIA Tesla V100
- (Planned) Export to ONNX/TensorRT for optimized real-time inference

---

## 🗂️ Directory Structure

├── datasets/ # Not pushed to GitHub (ignored)
│ ├── nuscenes/ # Raw NuScenes data (v1.0-mini)
│ └── yolo-format/ # YOLOv8-compatible structure
├── models/ # Exported weights (.pt)
├── runs/ # YOLOv8 training logs
├── scripts/
│ └── convert_nuscenes_to_yolo.py # NuScenes → YOLO format
├── yolo_config/
│ └── nuscenes.yaml # YOLO training config file
├── requirements.txt
├── README.md
└── .gitignore

yaml
Copy

---

## 🧠 Dataset

Using [NuScenes](https://www.nuscenes.org/) mini version:
- 10 training scenes, 2 validation scenes
- Converted using custom Python script
- Uses only the `CAM_FRONT` camera view
- Supports 10 object classes:
  - car, pedestrian, truck, bus, motorcycle, bicycle, traffic_cone, barrier, construction_vehicle, trailer

---

## 🚀 Training YOLOv8

First, activate your virtual environment:

```bash
source env/bin/activate
Then run:

bash
Copy
yolo detect train \
  data=yolo_config/nuscenes.yaml \
  model=yolov8s.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  amp=True \
  name=yolov8_nuscenes
🛠️ Preprocessing Script
The script scripts/convert_nuscenes_to_yolo.py converts NuScenes JSON annotations to YOLO .txt format. It also copies images into the images/train and images/val folders.

bash
Copy
cd scripts
python convert_nuscenes_to_yolo.py
📊 Evaluation
After training:

bash
Copy
yolo detect val model=runs/detect/yolov8_nuscenes/weights/best.pt data=yolo_config/nuscenes.yaml
Visuals like confusion_matrix.png, results.png are saved in runs/detect/yolov8_nuscenes/.

🔄 Roadmap
 Build preprocessing pipeline (NuScenes → YOLO)

 Train YOLOv8s on NuScenes mini

 Export to ONNX and TensorRT

 Run real-time inference on live video or dashcam feed

 (Optional) Add tracking module (DeepSORT)

📦 Dependencies
Install all dependencies:

bash
Copy
pip install -r requirements.txt
Main tools:

PyTorch

Ultralytics YOLOv8

OpenCV

NuScenes-devkit

PyQuaternion

👤 Author
Mohammed Harfan Abdul Azeez
Master of Engineering in Computer Vision & AI
University of Limerick
iamharfan/github.com
