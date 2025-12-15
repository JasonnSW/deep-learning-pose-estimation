# Deep Learning Pose Ergonomics Analysis

This project implements a **human pose ergonomics analysis system** using a **pretrained YOLOv8 Pose Estimation model**.  
The application detects human body keypoints from video input and evaluates ergonomic risk levels using a **rule-based REBA-lite approach**.

---

## Project Overview

- Input: Video file (e.g. office CCTV footage)
- Output:
  - Human bounding boxes
  - Body keypoints (COCO format)
  - Ergonomic risk classification:
    - Low Risk
    - Medium Risk
    - High Risk
- Model: YOLOv8 Pose (pretrained, COCO)

---

## Project Structure

deep-learning-pose-estimation/
├── src/
│ ├── main.py
│ ├── pose/
│ │ └── ergonomics.py
│ ├── ergonomics/
│ │ ├── angles.py
│ │ └── reba.py
│ └── utils/
│ └── draw.py
├── data/
│ └── videos/
│ └── test.mp4
├── yolov8s-pose.pt
├── requirements.txt
└── README.md


---

## Requirements

- **Python 3.13.x (MANDATORY)**
- pip
- Windows or Linux OS
- Video input (`.mp4`)

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/JasonnSW/deep-learning-pose-estimation.git
cd deep-learning-pose-estimation
```
### 2. Check Python Version
python --version

### 3. Install Dependencies
pip install -r requirements.txt


Running the Application
py src/main.py
