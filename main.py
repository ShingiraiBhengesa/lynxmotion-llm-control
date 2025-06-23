"""Main loop for robotic arm control using LLM and vision."""

import cv2
import argparse
import time
import yaml
import os
from vision.camera import LogitechCamera
from vision.object_detector import ObjectDetector
from arm_control.kinematics import calculate_ik
from arm_control.arduino_controller import ArduinoController
from llm.interface import LLMController
from utils.safety import check_joint_limits, validate_position

def main():
    parser = argparse.ArgumentParser(description="LLM Robotic Arm Controller")
    parser.add_argument('--port', default='COM5', help='Arduino serial port')
    args = parser.parse_args()

    # Load arm config for IK adjustments
    config_path = os.getenv('ARM_CONFIG_PATH', 'arm/arm_config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize components
    camera = LogitechCamera()
    detector = ObjectDetector(debug=True)
    llm = LLMController()
    arm = ArduinoController(port=args.port)

    print("🤖 LLM Robotic Arm Controller Started")
    arm.home_position()

    try:
        while True:
            # Capture frame
            ret, frame = camera.capture_frame()
            if not ret:
                print("❌ Camera frame capture failed.")
                break

            # Detect objects
            detected_objects = detector.detect_objects(frame)
            for obj in detected_objects:
                print(f"🟢 {obj['label']} at {obj['center_mm']} mm")

            # Get user command
            user_command = input("💬 Enter command (or 'exit'): ")
            if user_command.lower() == 'exit':
                break

            # Prepare LLM input
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

            prompt = f"""
You are controlling a 5-DOF robotic arm in a real workspace.

Objects detected:
{objects_seen}

User command:
"{user_command}"

Return a JSON command:
- Movement: {{"command": "MOVE", "target": [x,y,z], "speed": "slow"|"normal"|"fast"}}
- Gripper: {{"command": "GRIP", "gripper": "open"|"close"}}

Use ONLY provided positions. Return ONE JSON object.
"""

            # Query LLM with timeout
            try:
                start_time = time.time()
                response = llm.ask(prompt, image=frame)
                if time.time() - start_time > 2.0:
                    print("⚠️ LLM timeout, skipping.")
                    continue
            except Exception as e:
                print(f"⚠️ LLM error: {e}, trying text-only...")
                response = llm.ask_text_only(prompt)

            # Validate response
            if not isinstance(response, dict) or "command" not in response:
                print("⚠️ Invalid LLM response format.")
                continue
            if "error" in response:
                print(f"⚠️ LLM error: {response['error']}")
                continue

            if response["command"] == "MOVE":
                if not (isinstance(response.get("target"), list) and len(response["target"]) == 3):
                    print("⚠️ Invalid MOVE target.")
                    continue
                x, y, z = response["target"]
                if not validate_position(x, y, z):
                    print("❌ Target outside safe workspace.")
                    continue

                # Try IK with default gripper angle
                joint_angles = calculate_ik(x, y, z, grip_angle_d=90.0)
                if not joint_angles:
                    # Adjust z to account for base and gripper
                    adjusted_z = z + config['base_height'] + config['wrist_length']
                    print(f"⚠️ Retrying with adjusted z={adjusted_z:.2f}mm")
                    joint_angles = calculate_ik(x, y, adjusted_z, grip_angle_d=90.0)
                    if not joint_angles:
                        print("❌ Target unreachable with downward gripper.")
                        continue

                # Map speed to duration
                speed = response.get("speed", "normal")
                duration_map = {"slow": 4.0, "normal": 2.0, "fast": 1.0}
                duration = duration_map.get(speed, 2.0)

                # Validate joint angles before moving
                if not check_joint_limits(joint_angles):
                    print("❌ Joint limits exceeded. Command aborted.")
                    continue

                print(f"🎯 Moving to: ({x}, {y}, {z}) at {speed} speed.")
                arm.move_to(joint_angles, duration=duration)
                print("✅ Arm moved successfully.")

            elif response["command"] == "GRIP":
                action = response.get("gripper")
                if action not in ["open", "close"]:
                    print("⚠️ Invalid GRIPPER action.")
                    continue
                arm.control_gripper(action)
                print(f"✅ Gripper {action}.")

            else:
                print("⚠️ Unrecognized command.")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting...")

    finally:
        camera.release()
        arm.emergency_stop()
        arm.close()
        cv2.destroyAllWindows()
        print("👋 Shutdown complete.")

if __name__ == "__main__":
    main()
