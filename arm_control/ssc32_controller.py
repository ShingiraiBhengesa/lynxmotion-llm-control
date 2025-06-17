"""
SSC-32U controller for Lynxmotion robotic arm.

Sends servo angle commands to the SSC-32U over a serial port.
"""

import serial
import time
import json
import os

class SSC32Controller:
    def __init__(self, port, baudrate=115200):
        """
        Initializes the serial connection to the SSC-32U controller.
        
        Args:
            port (str): The serial port the controller is connected to (e.g., 'COM4' or '/dev/ttyUSB0').
            baudrate (int): The communication speed, which should match the controller's setting.
        """
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.joint_limits = self._load_joint_limits()
        print("✅ SSC-32U Controller Initialized.")
        
    def _load_joint_limits(self):
        """Loads joint angle limits from the JSON configuration file."""
        config_path = os.path.join(os.path.dirname(__file__), '../config/joint_limits.json')
        with open(config_path) as f:
            return json.load(f)
    
    def _validate_angles(self, angles):
        """
        Checks if the requested angles are within the safe limits defined in joint_limits.json.
        
        Args:
            angles (dict): A dictionary of {'joint_name': angle}.
            
        Returns:
            bool: True if all angles are within safe limits, False otherwise.
        """
        for joint, angle in angles.items():
            if joint in self.joint_limits:
                low, high = self.joint_limits[joint]
                if not low <= angle <= high:
                    print(f"⚠️ SAFETY WARNING: {joint} angle {angle}° is out of the safe range [{low}, {high}].")
                    return False
        return True
    
    def send_command(self, angles, move_time=1000):
        """
        Converts angles to pulse widths and sends the command string to the SSC-32U.
        
        Args:
            angles (dict): A dictionary of joint angles to move to (e.g., {'base': 90, 'shoulder': 120}).
            move_time (int): The time in milliseconds for the move to complete.
        """
        if not self._validate_angles(angles):
            print("❌ Command aborted due to safety limit violation.")
            return
            
        cmd = []
        for joint, angle in angles.items():
            # Convert angle (0-180) to servo pulse width (500-2500)
            pulse = int(500 + (2000 * (angle / 180.0)))
            channel = self._get_channel(joint)
            
            if channel != -1:
                cmd.append(f"#{channel}P{pulse}")
        
        if not cmd:
            print("🤷 No valid joint commands to send.")
            return

        # Append the time for the move and the carriage return terminator
        full_command = " ".join(cmd) + f"T{move_time}\r"
        
        print(f"➡️ Sending command: {full_command.strip()}")
        self.ser.write(full_command.encode())
    
    def emergency_stop(self):
        """Sends the STOP command to halt all servo movement immediately."""
        self.ser.write(b"STOP\r")
        print("🛑 EMERGENCY STOP ACTIVATED.")
    
    def home_position(self):
        """Moves the arm to a predefined home/neutral position."""
        print("🏠 Moving to home position...")
        self.send_command({
            'base': 90,
            'shoulder': 90,
            'elbow': 90,
            'wrist': 90
        })
    
    def _get_channel(self, joint):
        """
        Maps a joint name string to its physical channel number on the SSC-32U board.
        
        Args:
            joint (str): The name of the joint (e.g., 'base').
            
        Returns:
            int: The channel number, or -1 if not found.
        """
        # --- Channel map updated to match the working MATLAB configuration ---
        channel_map = {
            'base': 1,
            'shoulder': 2,
            'elbow': 3,
            'wrist': 4,
            'gripper': 5
        }
        
        channel = channel_map.get(joint)
        if channel is None:
            print(f"🤔 Warning: Joint '{joint}' not found in channel map.")
            return -1
        return channel

    def close(self):
        """Closes the serial port connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("🔌 Serial port closed.")