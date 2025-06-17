import cv2
import time
import os
import json
from arm_control.ssc32_controller import SSC32Controller
from arm_control.kinematics import calculate_ik
from vision.camera import LogitechCamera
from llm.interface import LLMController
from utils.safety import validate_position

# Configuration
ARM_PORT = '/dev/ttyUSB0'  # Update for your system
CAMERA_INDEX = 0
TEMP_IMAGE_PATH = "current_view.jpg"
DEBUG_MODE = True

def save_debug_image(frame, command, llm_response):
    """Save annotated image for debugging"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"debug/debug_{timestamp}.jpg"
    cv2.putText(frame, f"Command: {command}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"LLM: {llm_response}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(filename, frame)

def main():
    print("Initializing Lynxmotion Arm Control System...")
    
    # Initialize systems
    arm = SSC32Controller(ARM_PORT)
    camera = LogitechCamera(CAMERA_INDEX)
    llm = LLMController()
    
    # Create debug directory
    if DEBUG_MODE and not os.path.exists("debug"):
        os.makedirs("debug")
    
    # Move to home position
    print("Moving to home position...")
    arm.home_position()
    time.sleep(2)
    
    try:
        while True:
            # Get user input
            try:
                user_input = input("\nEnter command (or 'quit'): ")
            except EOFError:
                break
                
            if user_input.lower() in ['quit', 'exit']:
                break
            
            # Capture image
            ret, frame = camera.capture_frame()
            if not ret:
                print("Camera error! Check connection.")
                continue
            
            # Save temporary image
            cv2.imwrite(TEMP_IMAGE_PATH, frame)
            
            # Process with LLM
            print("Querying LLM...")
            command = llm.generate_command(user_input, os.path.abspath(TEMP_IMAGE_PATH))
            
            if DEBUG_MODE:
                print("LLM Response:", json.dumps(command, indent=2))
                save_debug_image(frame.copy(), user_input, json.dumps(command))
            
            # Execute command
            if "error" in command:
                print(f"LLM error: {command['error']}")
            elif command["command"] == "MOVE":
                try:
                    x, y, z = command["target"]
                    if not validate_position(x, y, z):
                        print(f"Invalid position: ({x}, {y}, {z})")
                        continue
                    print(f"Moving to: ({x}, {y}, {z})")
                    angles = calculate_ik(x, y, z)
                    print(f"Calculated angles: {angles}")
                    arm.send_command(angles)
                except Exception as e:
                    print(f"Movement error: {str(e)}")
            elif command["command"] == "GRIP":
                state = command["gripper"]
                print(f"Setting gripper to: {state}")
                arm.set_gripper(state)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        arm.emergency_stop()
        arm.close()
        camera.release()
        print("Systems shut down")

if __name__ == "__main__":
    main()