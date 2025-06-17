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

# ❗ CRITICAL: Update this with the COM port from your Windows Device Manager.
# For example, if your robot is on COM4, change it to: ARM_PORT = 'COM4'
ARM_PORT = 'COM4'  # Update this to your SSC-32U port

CAMERA_INDEX = 0
TEMP_IMAGE_PATH = "current_view.jpg"
DEBUG_MODE = True
RESOLUTION = (1280, 720)

def save_debug_image(frame, command, llm_response):
    """Save debug image with LLM response overlaid."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"debug/debug_{timestamp}.jpg"
    # Ensure the debug directory exists
    if not os.path.exists("debug"):
        os.makedirs("debug")
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

    # Initialize all components
    try:
        arm = SSC32Controller(ARM_PORT)
        camera = LogitechCamera(CAMERA_INDEX, RESOLUTION)
        detector = ObjectDetector()
        llm = LLMController()
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        print("   Please ensure all hardware is connected and configurations (like ARM_PORT) are correct.")
        return

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
                print("❌ Camera error. Could not capture frame.")
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

            if command.get("command") == "MOVE":
                x, y, z = command["target"]
                if not validate_position(x, y, z):
                    print(f"⚠️ SAFETY: Target position ({x}, {y}, {z}) is outside the valid workspace.")
                    continue
                
                angles = calculate_ik(x, y, z)
                if angles is None:
                    print(f"⚠️ KINEMATICS: Could not calculate a valid solution for position ({x}, {y}, {z}). It may be unreachable.")
                    continue
                
                if not check_joint_limits(angles):
                    print("⚠️ SAFETY: Calculated angles exceed joint limits.")
                    continue
                
                arm.send_command(angles)

            elif command.get("command") == "GRIP":
                # Note: The provided SSC32Controller doesn't have a 'set_gripper' method.
                # You would need to implement gripper control logic.
                # For now, we will print the intended action.
                print(f" gripper action: {command.get('gripper')}")


    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Stopping...")

    finally:
        # Gracefully shut down all components
        print("shutting down...")
        if 'arm' in locals() and arm:
            arm.emergency_stop()
            arm.close()
        if 'camera' in locals() and camera:
            camera.release()
        print("✅ Shutdown complete.")


if __name__ == "__main__":
    main()