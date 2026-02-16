# 🧠 Real-Time Face Detection using OpenCV DNN

## 📌 Overview

This project implements a **real-time face detection system** using OpenCV’s Deep Neural Network (DNN) module with a pre-trained **Caffe SSD ResNet-10 model**. It captures live video from a webcam, detects faces with confidence scores, and automatically saves detected faces.

The code is written in a **modular and production-ready manner**, making it easy to extend for real-world applications like surveillance, attendance systems, and security solutions.

---

## 🚀 Features

* 🎥 Real-time face detection using webcam
* 🧠 Deep learning-based detection (Caffe SSD model)
* 📊 Displays:

  * Face count
  * Detection confidence
  * FPS (Frames Per Second)
* 💾 Automatically saves detected face images
* ⚙️ Modular and clean code structure
* 🛑 Robust error handling

---

## 🏗️ Project Structure

```
project/
│── models/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
│── saved_faces/
│
│── face_detection.py
```

---

## ⚙️ Technologies Used

* Python
* OpenCV (cv2)
* NumPy
* Caffe Deep Learning Model

---

## ▶️ How It Works

1. Loads the pre-trained Caffe face detection model
2. Initializes webcam stream
3. Processes each frame using DNN
4. Detects faces based on confidence threshold
5. Draws bounding boxes and labels
6. Saves detected face images
7. Displays real-time FPS and face count

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/ankeshbit/Face_count.git
cd Face_count
```

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## 📥 Model Files Setup

Download the following files and place them inside the `models/` folder:

* `deploy.prototxt`
* `res10_300x300_ssd_iter_140000.caffemodel`

---

## ▶️ Usage

Run the application:

```bash
python face_detection.py
```

Press **`q`** to exit.

---

## 📸 Output

* Live webcam feed with:

  * Face bounding boxes
  * Confidence percentage
  * Face count
  * FPS
* Detected faces saved in `saved_faces/`

---

## ⚠️ Requirements

* Webcam (built-in or external)
* Python 3.x

---

## 🔥 Future Improvements

* Face recognition (identify individuals)
* Database integration
* Web-based dashboard
* Real-time alerts system

---

## 👨‍💻 Author

**Your Name**

---

## ⭐ Contributing

Contributions are welcome! Feel free to fork this repo and submit a pull request.

---
