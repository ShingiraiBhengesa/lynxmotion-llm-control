"""SSC-32U controller for Lynxmotion robotic arm.

Sends servo angle commands to the SSC-32U over a serial port.
"""

import serial
import time
import json
import os

class SSC32Controller:
    """Low-level servo controller using SSC-32U serial commands."""

    def __init__(self, port, baudrate=115200):
        """
        Args:
            port (str): Serial port (e.g., '/dev/ttyUSB0' or 'COM3').
            baudrate (int): Baud rate (default 115200).
        """
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.joint_limits = self._load_joint_limits()
        self.channel_map = {
            'base': 0,
            'shoulder': 1,
            'elbow': 2,
            'wrist': 3,
            'gripper': 4
        }

    def _load_joint_limits(self):
        """Load joint angle limits from JSON config file."""
        path = os.path.join(os.path.dirname(__file__), '../config/joint_limits.json')
        with open(path) as f:
            return json.load(f)

    def _validate_angles(self, angles):
        """Ensure joint angles are within safe limits."""
        for joint, angle in angles.items():
            if joint in self.joint_limits:
                low, high = self.joint_limits[joint]
                if not low <= angle <= high:
                    raise ValueError(f"{joint} angle {angle}° out of range [{low}, {high}]")

    def _angle_to_pulse(self, angle):
        """Convert an angle (degrees) to SSC-32U-compatible pulse (μs)."""
        return int(500 + 2000 * (angle / 180.0))  # Range: 500–2500 μs

    def send_command(self, angles, move_time=1000, speed=150):
        """Send servo angles to SSC-32U.
        
        Args:
            angles (dict): Joint-to-angle mapping.
            move_time (int): Duration of move in milliseconds.
            speed (int): Optional movement speed (0–255).
        """
        self._validate_angles(angles)

        cmd = ""
        for joint, angle in angles.items():
            if joint in self.channel_map:
                pulse = self._angle_to_pulse(angle)
                channel = self.channel_map[joint]
                cmd += f"#{channel}P{pulse}S{speed} "
        cmd += f"T{move_time}\r"

        self.ser.write(cmd.encode())
        time.sleep(move_time / 1000 + 0.1)

    def set_gripper(self, state):
        """Open or close the gripper.
        
        Args:
            state (str): 'open' or 'close'
        """
        angle = 0 if state == "open" else 100
        self.send_command({'gripper': angle}, move_time=500)

    def home_position(self):
        """Send robot to neutral home position."""
        self.send_command({
            'base': 90,
            'shoulder': 90,
            'elbow': 90,
            'wrist': 90,
            'gripper': 0
        })

    def emergency_stop(self):
        """Immediately stop all servo movement."""
        self.ser.write(b"STOP\r")

    def close(self):
        """Close the serial port cleanly."""
        self.ser.close()

    def __del__(self):
        """Destructor ensures cleanup if forgotten."""
        self.close()
