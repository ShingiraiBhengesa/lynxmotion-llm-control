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
        Initializes the serial connection and the arm's current state.
        """
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        # Store the current angles of the arm. Initialize to a neutral home position.
        self.current_angles = {'base': 90, 'shoulder': 90, 'elbow': 90, 'wrist': 90, 'gripper': 90}
        print("✅ Arduino Controller Initialized.")

    def _get_channel_id(self, joint_name):
        """
        Maps a joint name to its servo ID.
        """
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
        Moves the arm smoothly from current to target angles over a duration.

        Args:
            target_angles (dict): A dictionary of target joint angles.
            duration (float): The time in seconds for the movement.
        """
        print(f"🤖 Starting smooth move to {target_angles} over {duration}s...")
        # Determine the number of steps for a smooth motion (e.g., 25 steps per second)
        steps = int(duration * 25)
        if steps == 0:
            steps = 1

        # Get a copy of the starting angles from the arm's current state
        start_angles = self.current_angles.copy()

        for step in range(1, steps + 1):
            interpolated_angles = {}
            for joint in target_angles:
                if joint in start_angles:
                    start_angle = start_angles[joint]
                    target_angle = target_angles[joint]
                    
                    # Linear interpolation formula to find the angle for the current step
                    interpolated_angle = start_angle + (target_angle - start_angle) * (step / steps)
                    interpolated_angles[joint] = interpolated_angle
            
            # Send the calculated intermediate angles to the arm
            self._send_raw_command(interpolated_angles)
            # Wait for a small amount of time before the next step
            time.sleep(duration / steps)

        # Update the controller's state to reflect the arm's new position
        self.current_angles.update(target_angles)
        print("✅ Move complete.")

    def _send_raw_command(self, angles):
        """
        Sends individual angle commands to the Arduino.
        """
        for joint, angle in angles.items():
            servo_id = self._get_channel_id(joint)
            if servo_id:
                # Ensure the angle is within a safe physical range (e.g., 0-180) before sending
                safe_angle = max(0, min(180, angle))
                angle_val = int(round(safe_angle * 10))
                command = f"#{servo_id}D{angle_val}\r"
                
                # Uncomment the line below for verbose debugging of raw commands
                # print(f"➡️ Sending raw command: {command.strip()}")
                self.ser.write(command.encode())
        # A very small delay to ensure serial buffer doesn't overflow
        time.sleep(0.01)

    def home_position(self):
        """Moves the arm to a predefined home/neutral position smoothly."""
        print("🏠 Moving to home position...")
        home_angles = {
            'base': 90,
            'shoulder': 90,
            'elbow': 90,
            'wrist': 90,
            'gripper': 90
        }
        self.move_to(home_angles, duration=2.0)

    def close(self):
        """Closes the serial port connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("🔌 Serial port closed.")

    def emergency_stop(self):
        # This function is kept for compatibility with the main loop.
        # It does not send commands but stops the Python script.
        print("🛑 Emergency stop called (no command sent).")