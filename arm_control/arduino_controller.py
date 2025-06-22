"""
An Arduino-based controller for a Lynxmotion robotic arm.

Sends commands in the format '#<ID>D<Angle*10>\r' to an Arduino
running a custom sketch that understands this protocol.
"""
import serial
import time

class ArduinoController:
    def __init__(self, port, baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        self.current_angles = {'base': 90, 'shoulder': 90, 'elbow': 90, 'wrist': 90, 'gripper': 90}
        print("✅ Arduino Controller Initialized.")

    def _get_channel_id(self, joint_name):
        id_map = {
            'base': 1,
            'shoulder': 2,
            'elbow': 3,
            'wrist': 4,
            'gripper': 5
        }
        return id_map.get(joint_name)

    def move_to(self, target_angles, duration=2.0):
        print(f"🤖 Starting smooth move to {target_angles} over {duration}s...")
        steps = int(duration * 25)
        if steps == 0:
            steps = 1
        start_angles = self.current_angles.copy()
        for step in range(1, steps + 1):
            interpolated_angles = {}
            for joint in target_angles:
                if joint in start_angles:
                    start_angle = start_angles[joint]
                    target_angle = target_angles[joint]
                    interpolated_angle = start_angle + (target_angle - start_angle) * (step / steps)
                    interpolated_angles[joint] = interpolated_angle
            self._send_raw_command(interpolated_angles)
            time.sleep(duration / steps)
        self.current_angles.update(target_angles)
        print("✅ Move complete.")

    def _send_raw_command(self, angles):
        for joint, angle in angles.items():
            servo_id = self._get_channel_id(joint)
            if servo_id:
                safe_angle = max(0, min(180, angle))
                angle_val = int(round(safe_angle * 10))
                command = f"#{servo_id}D{angle_val}\r"
                self.ser.write(command.encode())
        time.sleep(0.01)

    def home_position(self):
        print("🏠 Moving to home position...")
        home_angles = {'base': 90, 'shoulder': 90, 'elbow': 90, 'wrist': 90, 'gripper': 90}
        self.move_to(home_angles, duration=2.0)

    def control_gripper(self, action, duration=0.5):
        """
        Controls the gripper with a smooth motion.
        """
        if action == "open":
            print("🖐️ Opening gripper...")
            self.move_to({'gripper': 0}, duration=duration)
        elif action == "close":
            print("✊ Closing gripper...")
            self.move_to({'gripper': 100}, duration=duration)
        else:
            print(f"⚠️ Unknown gripper action: {action}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("🔌 Serial port closed.")

    def emergency_stop(self):
        print("🛑 Emergency stop called (no command sent).")