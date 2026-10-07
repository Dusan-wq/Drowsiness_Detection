# Heuristic Drowsiness Detection

Real-time driver drowsiness detection system based on heuristic computer vision methods and YOLO object detection.

## Overview

The system monitors a driver through a camera feed and detects signs of drowsiness such as prolonged eye closure, yawning, and head nodding. Detection is based on heuristic rules applied to facial landmarks and object detection results, without requiring a dedicated trained classifier for the drowsiness state itself.

Authors: Nemanja Mitrović, Stefan Vasić, Ognjen Ikrašev, Dušan Vukanić
Professor: Vladimir Bugarski, PhD
## Features

- Real-time face and eye detection
- Eye Aspect Ratio (EAR) based drowsiness detection
- Yawning detection via Mouth Aspect Ratio (MAR)
- Head pose estimation
- Alarm/alert when drowsiness is detected
- Works with a standard webcam

## Technologies

- Python
- OpenCV
- YOLO (Ultralytics)
- NumPy
- dlib / MediaPipe (facial landmarks)

## Setup

1. Clone the repository:
   git clone https://github.com/Dusan-wq/Drowsiness_Detection.git
   cd Drowsiness_Detection

2. Install dependencies:
   pip install -r requirements.txt

3. Download the YOLO model (yolov8s.pt) if not included.

## Usage

   python main.py

The webcam feed will open. The system will track the driver's face and trigger an alert if signs of drowsiness are detected.

## How It Works

1. Face detection — YOLO or a Haar cascade locates the driver's face in each frame.
2. Landmark extraction — facial landmarks (eyes, mouth) are extracted.
3. Heuristic rules:
   - If EAR (Eye Aspect Ratio) drops below a threshold for N consecutive frames -> eyes are closing -> drowsiness.
   - If MAR (Mouth Aspect Ratio) exceeds a threshold -> yawning.
   - Head tilt / nodding detection via landmark geometry.
4. Alert — when drowsiness is confirmed, an alarm sound and/or visual warning is triggered.

## Results

The system runs in real time on a standard CPU and detects drowsiness within a few seconds of the first signs.
