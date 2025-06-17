# Lynxmotion Robotic Arm Control with LLM and Vision

This project enables natural language control of a Lynxmotion robotic arm using a custom Arduino-based controller, a Large Language Model (OpenAI's GPT-4), and real-time computer vision.

The system interprets commands like "pick up the red block" by analyzing a live camera feed and generating the correct robotic movements.

## Features

-   Voice/text command processing via LLM (OpenAI GPT-4 Vision)
-   Real-time object detection with YOLOv8
-   Inverse kinematics for a 5-DOF robotic arm
-   Control of an Arduino-based servo controller
-   Safety checks for workspace and joint limits
-   Visual debugging system

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
git clone [https://github.com/yourusername/lynxmotion-llm-control.git](https://github.com/yourusername/lynxmotion-llm-control.git)
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

```OPENAI_API_KEY=your_secret_api_key_here
```


