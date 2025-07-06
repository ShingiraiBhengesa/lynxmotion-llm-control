"""
An Arduino-based controller for a Lynxmotion robotic arm.

Sends commands in the format '#<ID>D<Angle*10>\r' to an Arduino
running a custom sketch that understands this protocol.
"""
import serial
import time
from utils.safety import check_joint_limits

class ArduinoController:
    def __init__(self, port, baudrate=115200):
        """
        Initialize serial connection to Arduino.

        Args:
            port (str): Serial port (e.g., 'COM5').
            baudrate (int): Baud rate (default 115200).
        """
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            self.ser.flush()  # Clear buffers
            time.sleep(2)  # Wait for Arduino to initialize
            self.current_angles = {'base': 90, 'shoulder': 90, 'elbow': 90, 'wrist': 90, 'gripper': 90}
            print(f"✅ Arduino Controller Initialized on {port}")
        except serial.SerialException as e:
            raise RuntimeError(f"Failed to open serial port {port}: {e}")

    def _get_channel_id(self, joint_name):
        """Map joint names to servo IDs."""
        id_map = {
            'base': 1,
            'shoulder': 2,
            'elbow': 3,
            'wrist': 4,
            'gripper': 5
        }
        return id_map.get(joint_name)

    def move_to(self, target_angles, duration=2.0):
        """
        Smoothly move to target joint angles.

        Args:
            target_angles (dict): Joint angles in degrees.
            duration (float): Movement duration in seconds.
        """
        if duration <= 0:
            raise ValueError("Duration must be positive")
        if not check_joint_limits(target_angles):
            print("❌ Move aborted: Joint angles outside safe limits.")
            return
        print(f"🤖 Moving to {target_angles} over {duration}s...")
        steps = max(1, int(duration * 25))  # Ensure at least 1 step
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
        """Send raw servo commands to Arduino."""
        try:
            for joint, angle in angles.items():
                servo_id = self._get_channel_id(joint)
                if servo_id:
                    # Clamp to [0, 180] for non-gripper joints
                    safe_angle = angle if joint == 'gripper' else max(0, min(180, angle))
                    angle_val = int(round(safe_angle * 10))
                    command = f"#{servo_id}D{angle_val}\r"
                    self.ser.write(command.encode())
                    self.ser.flush()  # Ensure command is sent
            time.sleep(0.01)  # Brief pause for Arduino processing
        except serial.SerialException as e:
            print(f"⚠️ Serial communication error: {e}")

    def home_position(self):
        """Move to home position (all joints at 90°)."""
        print("🏠 Moving to home position...")
        home_angles = {'base': 90, 'shoulder': 90, 'elbow': 90, 'wrist': 90, 'gripper': 90}
        self.move_to(home_angles, duration=2.0)

    def control_gripper(self, action, duration=0.5):
        """
        Control the gripper with smooth motion.

        Args:
            action (str): 'open' or 'close'.
            duration (float): Movement duration in seconds.
        """
        if action == "open":
            print("🖐️ Opening gripper...")
            self.move_to({'gripper': 0}, duration=duration)
        elif action == "close":
            print("✊ Closing gripper...")
            self.move_to({'gripper': 100}, duration=duration)
        else:
            print(f"⚠️ Unknown gripper action: {action}")

    def emergency_stop(self):
        """Send emergency stop to all servos."""
        print("🛑 Emergency stop called.")
        try:
            # Send neutral position commands instead of 'S' (safer fallback)
            home_angles = {'base': 90, 'shoulder': 90, 'elbow': 90, 'wrist': 90, 'gripper': 90}
            self._send_raw_command(home_angles)
            self.ser.flush()
            time.sleep(0.1)
        except serial.SerialException as e:
            print(f"⚠️ Serial error during emergency stop: {e}")

    def close(self):
        """Close the serial port."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("🔌 Serial port closed.")

    def __del__(self):
        self.close()