# 🎓 AI Classroom Monitoring System

An intelligent classroom monitoring system that combines **Face Recognition**, **Emotion Detection**, and **Real-Time Analytics Dashboards** using Python and R.

---

## 📌 Overview

This project provides a smart way to monitor student engagement during lectures by:

- Detecting student faces using AI
- Recognizing identities from a dataset
- Analyzing facial emotions in real-time
- Logging data for further analysis
- Visualizing engagement using interactive dashboards

---

## 🧠 Features

- 🎥 Real-time face detection using OpenCV  
- 🧑‍🎓 Student recognition using DeepFace (FaceNet)  
- 😊 Emotion detection (Happy, Sad, Neutral, etc.)  
- 📊 Live analytics dashboard using R Shiny  
- 📈 Python dashboard for additional insights  
- 📝 Automatic logging to CSV file  
- 🌐 Accessible via local network  

---

## 🛠️ Tech Stack

### 🔹 Python
- OpenCV
- DeepFace
- Pandas

### 🔹 R
- Shiny
- ggplot2
- dplyr

---

## 📂 Project Structure
- ai-classroom/
│
├── ai_engine.py # Face recognition + emotion detection
├── dashboard.py # Python dashboard
├── app.R # R Shiny dashboard
├── students/ # Student images database
├── students.csv # Student IDs and names
├── emotion_log.csv # Generated data log
└── download_images.py # Dataset preparation script

---

## Run the Project

Open app.R and click Run in VS Code

OR manually:

library(shiny)
runApp("path/to/project")

## What Happens

Camera starts automatically 🎥
Face recognition begins
Emotion data is logged
Dashboards launch automatically
📊 Output Example
Student Name + Emotion + Confidence
Engagement score visualization
Attendance tracking

## ⚠️ Notes

Ensure student images are placed in the students/ folder
File names must match student IDs
Lighting conditions affect accuracy
