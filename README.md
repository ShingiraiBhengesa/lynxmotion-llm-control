# Lynxmotion Robotic Arm Control with LLM and Vision

This project enables natural language control of a Lynxmotion robotic arm (SSC-32U controller) using Large Language Models (LLMs) and computer vision.

![System Architecture](docs/system_architecture.png)

## Features

- Voice/text command processing via LLM (OpenAI GPT-4 Turbo)
- Real-time object detection with YOLOv8
- Camera-to-arm coordinate transformation
- Inverse kinematics for Lynxmotion AL5D arm
- Safety checks and collision avoidance
- Visual debugging system

## Hardware Requirements

1. Lynxmotion robotic arm with SSC-32U controller
2. Logitech webcam (C920 or similar)
3. Computer with USB ports
4. Power supply for robotic arm

## Software Requirements

- Python 3.8+
- VS Code (recommended)
- OpenAI API key

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/lynxmotion-llm-control.git
cd lynxmotion-llm-control

