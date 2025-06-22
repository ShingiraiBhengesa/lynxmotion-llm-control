"""Main loop for robotic arm control using LLM and vision."""

import cv2
from vision.object_detector import ObjectDetector
from arm_control.kinematics import calculate_ik
from arm_control.arduino_controller import ArduinoController
from llm.interface import LLMController
from utils.safety import check_joint_limits # <-- IMPORT FOR SAFETY CHECK

def main():
    # ✅ Initialize camera, detector, LLM, and robot arm
    camera = cv2.VideoCapture(0)
    detector = ObjectDetector(debug=True)
    llm = LLMController()
    # IMPORTANT: Make sure the serial port is correct for your system (e.g., 'COM3' on Windows)
    arm = ArduinoController(port='COM5') 

    print("🤖 LLM Robotic Arm Controller Started")
    arm.home_position()

    try:
        while True:
            # 🎥 Capture frame
            ret, frame = camera.read()
            if not ret:
                print("❌ Camera frame capture failed.")
                break

            # 🔍 Detect objects using OpenCV + camera calibration
            detected_objects = detector.detect_objects(frame)
            for obj in detected_objects:
                print(f"🟢 {obj['label']} at {obj['center_mm']} mm")

            # 🗣️ Take user input (text command)
            user_command = input("💬 Enter command (or 'exit'): ")
            if user_command.lower() == 'exit':
                break

            # 📦 Prepare LLM input format
            objects_seen = [
                {
                    "label": obj["label"],
                    "position": {
                        "x": round(obj["center_mm"][0], 2),
                        "y": round(obj["center_mm"][1], 2),
                        "z": round(obj["center_mm"][2], 2)
                    }
                }
                for obj in detected_objects
            ]

            # ✉️ Prompt sent to LLM (GPT-4 Vision if image is supported)
            prompt = f"""
You are controlling a 5-DOF robotic arm in a real workspace.

Here are the objects detected:
{objects_seen}

User command:
\"{user_command}\"

Choose the correct object and return a JSON command:

- For movement:
  {{ "command": "MOVE", "target": [x, y, z] }}
- For gripper control:
  {{ "command": "GRIP", "gripper": "open" or "close" }}

Use ONLY the object positions provided above. Do not guess.
Only return one JSON object. Do not include explanations.
"""

            print("📤 Sending prompt to LLM...")
            response = llm.ask(prompt, image=frame)
            print("🤖 LLM Response:", response)

            # ✅ Execute the LLM-decided action
            if response.get("command") == "MOVE":
                x, y, z = response["target"]
                
                # Get speed from LLM and map to a duration value
                speed = response.get("speed", "normal")
                duration_map = {"slow": 4.0, "normal": 2.0, "fast": 1.0}
                duration = duration_map.get(speed, 2.0)

                print(f"🎯 Moving to: {x}, {y}, {z} at {speed} speed.")
                joint_angles = calculate_ik(x, y, z)

                if joint_angles:
                    # Check joint limits before telling the arm to move
                    if check_joint_limits(joint_angles):
                        arm.move_to(joint_angles, duration=duration)
                        print("✅ Arm moved successfully.")
                    else:
                        print("❌ SAFETY: Move aborted. Calculated joint angles are outside safe limits.")
                else:
                    print("❌ Target unreachable or outside joint limits.")

            elif response.get("command") == "GRIP":
                action = response["gripper"]
                # Use the dedicated, cleaner gripper control method
                arm.control_gripper(action)

            else:
                print("⚠️ Invalid or unrecognized response from LLM.")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting...")

    finally:
        camera.release()
        arm.emergency_stop()
        cv2.destroyAllWindows()
        print("👋 Shutdown complete.")

if __name__ == "__main__":
    main()