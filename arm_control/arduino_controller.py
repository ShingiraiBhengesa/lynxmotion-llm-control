"""
An Arduino-based controller for a Lynxmotion robotic arm.

Sends commands in the format '#<ID>D<Angle*10>\r' to an Arduino
running a custom sketch that understands this protocol.
"""
import serial
import time

class ArduinoController:
    def __init__(self, port, baudrate=115200):
        """
        Initializes the serial connection. Baud rate is set to 115200.
        """
        self.ser = serial.Serial(port, baudrate, timeout=1)
        # Wait for the controller to be ready
        time.sleep(2)
        print("✅ Arduino Controller Initialized .")
        
    def _get_channel_id(self, joint_name):
        """
        Maps a joint name to its servo ID .
        """

        # (Channels 1, 2, 3, 4 for arm, 5 for gripper).
        id_map = {
            'base': 1,
            'shoulder': 2,
            'elbow': 3,
            'wrist': 4,
            'gripper': 5
        }
        return id_map.get(joint_name)

    def send_command(self, angles):
        """
        Sends individual angle commands.

        Args:
            angles (dict): A dictionary of joint angles {'joint_name': angle}.
        """
        for joint, angle in angles.items():
            servo_id = self._get_channel_id(joint)
            if servo_id:
                angle_val = int(round(angle * 10))
                command = f"#{servo_id}D{angle_val}\r"
                
                print(f"➡️ Sending command: {command.strip()}")
                self.ser.write(command.encode())
                # Small delay between commands to be safe
                time.sleep(0.05)
    
    def home_position(self):
        """Moves the arm to a predefined home/neutral position."""
        print("🏠 Moving to home position...")
        self.send_command({
            'base': 90,
            'shoulder': 90,
            'elbow': 90,
            'wrist': 90,
            'gripper': 90
        })

    def close(self):
        """Closes the serial port connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("🔌 Serial port closed.")

    def emergency_stop(self):
        # This function is kept for compatibility with the main loop.
        print("🛑 Emergency stop called (no command sent).")