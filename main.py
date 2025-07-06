import cv2
import argparse
import time
import yaml
import os
import torch
from vision.camera import LogitechCamera
from arm_control.kinematics import calculate_ik
from arm_control.arduino_controller import ArduinoController
from vlm_controller import VLMController
from utils.safety import check_joint_limits, validate_position

def main():
    parser = argparse.ArgumentParser(description="VLM Robotic Arm Controller")
    parser.add_argument('--port', default='COM5', help='Arduino serial port')
    args = parser.parse_args()

    config_path = os.getenv('ARM_CONFIG_PATH', 'config/arm_config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    camera = LogitechCamera(resolution=(640, 480))
    vlm = VLMController(debug=True)
    arm = ArduinoController(port=args.port)

    print("🤖 VLM Robotic Arm Controller Started")
    arm.home_position()

    try:
        last_detection_time = 0
        while True:
            ret, frame = camera.capture_frame()
            if not ret:
                print("❌ Camera frame capture failed.")
                break

            if time.time() - last_detection_time >= 2.0:
                start_time = time.time()
                detected_objects = vlm.detect_objects(frame)
                print(f"🟢 Detection took {time.time() - start_time:.2f}s")
                for obj in detected_objects:
                    print(f"🟢 {obj['label']} at {obj['center_mm']} mm")
                last_detection_time = time.time()

            user_command = input("💬 Enter command (or 'exit'): ")
            if user_command.lower() == 'exit':
                break

            start_time = time.time()
            response = vlm.plan_action(user_command, frame, detected_objects)
            print(f"🟢 Planning took {time.time() - start_time:.2f}s")

            if not isinstance(response, dict) or "command" not in response:
                print("⚠️ Invalid VLM response format.")
                continue
            if "error" in response:
                print(f"⚠️ VLM error: {response['error']}")
                continue

            if response["command"] == "MOVE":
                if not (isinstance(response.get("target"), list) and len(response["target"]) == 3):
                    print("⚠️ Invalid MOVE target.")
                    continue
                x, y, z = response["target"]
                if not validate_position(x, y, z):
                    print("❌ Target outside safe workspace.")
                    continue

                joint_angles = calculate_ik(x, y, z, grip_angle_d=90.0)
                if not joint_angles:
                    adjusted_z = z + config['base_height'] + config['wrist_length']
                    print(f"⚠️ Retrying with adjusted z={adjusted_z:.2f}mm")
                    joint_angles = calculate_ik(x, y, adjusted_z, grip_angle_d=90.0)
                    if not joint_angles:
                        print("❌ Target unreachable with downward gripper.")
                        continue

                speed = response.get("speed", "normal")
                duration_map = {"slow": 4.0, "normal": 2.0, "fast": 1.0}
                duration = duration_map.get(speed, 2.0)

                if not check_joint_limits(joint_angles):
                    print("❌ Joint limits exceeded. Command aborted.")
                    continue

                print(f"🎯 Moving to: ({x}, {y}, {z}) at {speed} speed.")
                arm.move_to(joint_angles, duration=duration)
                print("✅ Arm moved successfully.")
                time.sleep(1.0)

            elif response["command"] == "GRIP":
                action = response.get("gripper")
                if action not in ["open", "close"]:
                    print("⚠️ Invalid GRIPPER action.")
                    continue
                arm.control_gripper(action)
                print(f"✅ Gripper {action}.")
                time.sleep(1.0)

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