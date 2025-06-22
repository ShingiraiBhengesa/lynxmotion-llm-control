"""
An Arduino-based controller for a Lynxmotion robotic arm.

Sends commands in the format '#<ID>D<Angle*10>\r' to an Arduino
running a custom sketch that understands this protocol.
"""
import serial
import time

class ArduinoController:
    def __init__(self, port, baudrate=9600, timeout=2):
        self.servo_mapping = {
            "base": 1,
            "shoulder": 2,
            "elbow": 3,
            "wrist": 4,
            "gripper": 5
        }
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(2)  # Allow Arduino to reset
            print(f"✅ Connected to Arduino on {port}")
        except serial.SerialException as e:
            print(f"❌ Failed to connect to Arduino on {port}: {e}")
            self.ser = None

    def send_command(self, joint, angle):
        if self.ser is None:
            print("❌ Serial connection not initialized.")
            return

        if joint not in self.servo_mapping:
            print(f"❌ Unknown joint: {joint}")
            return

        channel = self.servo_mapping[joint]
        command = f"{channel}:{angle}\n"
        try:
            self.ser.write(command.encode())
            print(f"📤 Sent command: {command.strip()}")
        except serial.SerialException as e:
            print(f"❌ Failed to send command: {e}")

    def move_to(self, joint_angles: dict):
        """
        Move each servo to the specified angle.
        joint_angles: dict with joint names as keys and angles in degrees as values.
        """
        for joint, angle in joint_angles.items():
            self.send_command(joint, angle)

    def control_gripper(self, action: str):
        """
        Open or close the gripper.
        """
        if action.lower() == "open":
            self.send_command("gripper", 0)  # Open angle
        elif action.lower() == "close":
            self.send_command("gripper", 90)  # Close angle
        else:
            print(f"⚠️ Unknown gripper action: {action}")

    def home_position(self):
        print("🔄 Moving to home position")
        self.move_to({
            "base": 90,
            "shoulder": 90,
            "elbow": 90,
            "wrist": 90,
            "gripper": 0
        })

    def emergency_stop(self):
        if self.ser is None:
            print("❌ Serial connection not initialized.")
            return
        print("🛑 Emergency stop initiated")
        for joint in self.servo_mapping:
            self.send_command(joint, 90)  # Neutral angle for emergency stop

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            print("🔌 Serial connection closed")
