# 🦾 Robotic VLM-Control

Control a 5-DOF Lynxmotion robotic arm using natural language commands and vision-based object recognition powered by OpenCV and GPT-4 (Vision).

---

## Features

- **Real-time object detection** using OpenCV and color segmentation
- **3D pixel-to-world coordinate transformation** using chessboard camera calibration
- **Inverse kinematics** for 5-DOF Lynxmotion arm
- **Arduino serial control** via SSC-32U servo controller
- **LLM-powered natural language interface** with GPT-4 (Vision)
- **Safety limits** for robot joint angles and workspace bounds
- **Debug mode** with visual overlays and annotated logs

---
## Hardware Requirements

1.  **Lynxmotion Robotic Arm** with standard PWM servos.
2.  **Arduino-Compatible Controller:** An Arduino or similar microcontroller board capable of driving multiple servos.
3.  **Logitech Webcam:** A standard USB webcam (e.g., C920 or similar).
4.  **Computer with Python:** To run the main application.
5.  **Power Supply:** Appropriate power supply for the robotic arm servos.

## Software Requirements

-   Python 3.8+
-   An OpenAI API key

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ShingiraiBhengesa/vlm-robotic-control.git
cd lynxmotion-llm-control
```
### 2. Install Dependencies

```
# For Linux/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate

```

### 3. Install Dependencies

-  Install all the required Python libraries from the requirements.txt file.

```bash
pip install -r requirements.txt
```

### 4. Set Up OpenAI API Key

Create a file named ```.env``` in the main project directory. This file will securely store your API key.

```
OPENAI_API_KEY=your_secret_api_key_here
```


