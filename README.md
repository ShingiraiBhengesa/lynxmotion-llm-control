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
 ```

### 2. Create Virtial Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up OpenAI API Key: Create .env file

```bash
OPENAI_API_KEY=your_api_key_here
```

### 5. Hardware Configuration

- Hardware Configuration:
- Update ARM_PORT in  ```bash  main.py (e.g., /dev/ttyUSB0 or COM3)```

- Adjust arm dimensions in  ```bash arm_control/kinematics.py```

- Verify servo limits in  ```bash config/joint_limits.json```

