# 🦾 VLM Robotic Control

Control a 5-DOF Lynxmotion robotic arm using natural language commands, with object detection and task planning powered by the Florence-2 vision-language model (via Hugging Face `transformers`).

---

## Features

- **Object Detection and Task Planning**: Florence-2 vision-language model detects objects and interprets natural language commands (e.g., “move to red object”, “open gripper”) for task planning.
- **3D Pixel-to-World Coordinate Transformation**: Uses OpenCV for chessboard-based camera calibration and coordinate mapping.
- **Inverse Kinematics**: Computes joint angles for accurate 5-DOF Lynxmotion arm positioning.
- **Arduino Serial Control**: Interfaces with SSC-32U or compatible servo controller.
- **Natural Language Interface**: Processes commands via Florence-2 for intuitive control.
- **Safety Limits**: Enforces joint angle and workspace bounds to prevent collisions.
- **Debug Mode**: Saves visual overlays and logs in `debug_images/` for troubleshooting.

---

## Hardware Requirements

- **Lynxmotion Robotic Arm**: 5-DOF arm (e.g., AL5D) with standard PWM servos.
- **Arduino-Compatible Controller**: Arduino board with a sketch supporting the `#<ID>D<Angle*10>\r` protocol (e.g., `#1D900\r` for 90° on servo 1).
- **Logitech Webcam**: USB webcam (e.g., C920, 640x480 resolution recommended).
- **Computer**: Windows PC with Python 3.12.5 (CPU-based, no GPU required).
- **Power Supply**: 6V–7.4V for servos, per Lynxmotion specifications.

## Software Requirements

- Python 3.12.5
- Dependencies listed in `requirements.txt` (includes `transformers`, `torch` for CPU, `opencv-python`, etc.)
- Optional: OpenAI API key for alternative VLM models (not required for default Florence-2 setup)


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ShingiraiBhengesa/lynxmotion-llm-control.git
cd lynxmotion-llm-control
