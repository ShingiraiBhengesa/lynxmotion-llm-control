"""Main loop for robotic arm control using LLM and vision."""

import cv2
import time
import os
import json
from arm_control.ssc32_controller import SSC32Controller
from arm_control.kinematics import calculate_ik
from vision.camera import LogitechCamera
from vision.detector import ObjectDetector
from llm.interface import LLMController
from utils.safety import validate_position, check_joint_limits

# --- Configurable Constants ---
ARM_PORT = '/dev/ttyUSB0'   # Change to your serial port (e.g., COM3 on Windows)
CAMERA_INDEX = 0
TEMP_IMAGE_PATH = "current_view.jpg"
DEBUG_MODE = True
RESOLUTION = (1280, 720)

def save_debug_image(frame, command, llm_response):
    """Save debug image with LLM response overlaid."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"debug/debug_{timestamp}.jpg"
    cv2.putText(frame, f"Command: {command}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"LLM: {json.dumps(llm_response)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imwrite(filename, frame)

def main():
    """Main control loop for Lynxmotion arm."""
    print("🟢 Starting Lynxmotion LLM-Control System")

    if DEBUG_MODE and not os.path.exists("debug"):
        os.makedirs("debug")

    arm = SSC32Controller(ARM_PORT)
    camera = LogitechCamera(CAMERA_INDEX, RESOLUTION)
    detector = ObjectDetector()
    llm = LLMController()

    print("🏠 Moving to home position...")
    arm.home_position()
    time.sleep(2)

    try:
        while True:
            user_input = input("\nType a command (or 'quit'): ")
            if user_input.lower() in ['quit', 'exit']:
                break

            ret, frame = camera.capture_frame()
            if not ret:
                print("❌ Camera error.")
                continue

            cv2.imwrite(TEMP_IMAGE_PATH, frame)
            print("🤖 Querying LLM...")
            command = llm.generate_command(user_input, os.path.abspath(TEMP_IMAGE_PATH))

            if DEBUG_MODE:
                print("LLM Response:", json.dumps(command, indent=2))
                save_debug_image(frame.copy(), user_input, command)

            if "error" in command:
                print(f"LLM error: {command['error']}")
                continue

            if command["command"] == "MOVE":
                x, y, z = command["target"]
                if not validate_position(x, y, z):
                    print(f"⚠️ Invalid position: ({x}, {y}, {z})")
                    continue
                angles = calculate_ik(x, y, z)
                if not check_joint_limits(angles):
                    print("⚠️ Joint limit exceeded.")
                    continue
                arm.send_command(angles)

            elif command["command"] == "GRIP":
                arm.set_gripper(command["gripper"])

    except KeyboardInterrupt:
        print("🛑 Interrupted. Stopping...")

    finally:
        arm.emergency_stop()
        arm.close()
        camera.release()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
