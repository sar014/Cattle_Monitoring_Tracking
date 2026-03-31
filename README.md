# Cattle Monitoring & Tracking System

A computer vision-based system for **cow detection, tracking, counting, pose estimation, and behaviour monitoring** using **YOLOv8**, **YOLOv8-Pose**, **DeepSORT**, and **Random Forest**.

This project is designed to help monitor cattle activity through video analysis and can be extended for **livestock welfare monitoring**, **smart farming**, and **early distress detection**.

---

## 📌 Project Overview

This system processes cattle video footage and performs the following tasks:

- **Detects cows** in each frame
- **Tracks individual cows** across frames
- **Counts cattle** in the scene
- **Extracts body keypoints** using pose estimation
- **Classifies behaviour** such as:
  - Standing
  - Lying
  - Eating
- **Identifies possible signs of distress** based on posture / movement patterns

---

## Features

- **Cow Detection** using YOLOv8
- **Multi-object Tracking** using DeepSORT
- **Pose Estimation** using YOLOv8-Pose
- **Behaviour Classification** using Random Forest
- **Streamlit Web App** for easy video upload and inference
- Modular pipeline for future extension into:
  - real-time monitoring
  - health analysis
  - abnormal behaviour detection

---

## Tech Stack

- **Python**
- **YOLOv8**
- **YOLOv8-Pose**
- **DeepSORT**
- **OpenCV**
- **Scikit-learn**
- **Pandas / NumPy**
- **Streamlit**

---

## How it Works
The complete pipeline works in the following stages:

1. Cow Detection & Counting
    
    A YOLOv8 object detection model is used to detect cows in each frame. Bounding boxes are generated around each detected cow. The total number of visible cows can be counted frame by frame.

2. Cow Tracking
    
    DeepSORT is used to assign a unique ID to each detected cow. This allows the system to:
    * Track cows across frames
    * Avoid duplicate counting
    * Analyze behaviour for individual animals over time

3. Pose Estimation
    
    A YOLOv8-Pose model is used to detect cow body keypoints. These keypoints represent important body parts and help describe posture and movement.

4. Behaviour Classification
    
    Extracted pose/keypoint-based features are converted into structured data. A Random Forest classifier is used to classify behaviours such as:
    * Walking
    * Lying
    * Eating
    * In distress

5. Distress Monitoring
    
    Based on posture and behavioural patterns, the system can be extended to identify:
    * Prolonged inactivity
    * Unusual posture
    * Abnormal movement patterns
    * Potential distress indicators

## How to Run the Project
**Step 1: Extract the project files**
Extract all files from the Implementation folder to your local system.

**Step 2: Create a virtual environment**
```bash
python -m venv venv
```

**Step 3: Activate the environment**
```bash
venv\Scripts\activate
```

**Step 4: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 5: Run the Streamlit app**
```bash
python -m streamlit run app.py
```
**Step 6: Upload a video**
Open Streamlit app in browser and upload video to detect the behaviors.

## File Descriptions
1. Cattle_Detection_Counting
   
Contains code for:
* cow detection using YOLO
* cattle counting
* tracking using DeepSORT

2. Pose_Estimation
    
    Contains code to test pose estimation on cows.

    Note: This folder is for testing/inference only and not for training.

3. Detect+Pose
    
    A combined pipeline integrating:
    * detection
    * tracking
    * pose estimation

4. Cow_Random_Forest
   
    Contains code for training the Random Forest behaviour classification model.
    To train this model:
    * You need a CSV dataset similar to master_cow_dataV2.csv.
    * This CSV was created through the following pipeline:
    * Extract keypoints from pose estimation output in JSON format
    * Generate numerical features from the JSON data
    * Combine all extracted features into a final CSV file
    * Use this CSV to train the Random Forest classifier

## Datasets Used
1. Cattle Detection Dataset
Cattle Dataset (Pig, Sheep, Cow, Horse) – Kaggle 
https://www.kaggle.com/datasets/amiteshpatra07/cattle-dataset-pig-sheep-cow-horse

2. Cow Pose Estimation Dataset
Cow Pose Estimation Dataset – Kaggle
https://www.kaggle.com/datasets/zaidworks0508/cow-pose-estimation-dataset

## Important Note
The pose estimation model was trained using Roboflow, so the original training configuration files (such as the .yaml file) are not included in this repository.

This repository currently provides:
* inference/testing code
* integration pipeline
* behaviour classification workflow