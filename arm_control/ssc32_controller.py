"""SSC-32U controller for Lynxmotion robotic arm.

Sends servo angle commands to the SSC-32U over a serial port.
"""

import serial
import time
import json
import os

class SSC32Controller:
    def __init__(self, port, baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.joint_limits = self._load_joint_limits()
        
    def _load_joint_limits(self):
        config_path = os.path.join(os.path.dirname(__file__), '../config/joint_limits.json')
        with open(config_path) as f:
            return json.load(f)
    
    def _validate_angles(self, angles):
        for joint, angle in angles.items():
            low, high = self.joint_limits[joint]
            if not low <= angle <= high:
                raise ValueError(f"{joint} angle {angle}° out of range [{low}, {high}]")
    
    def send_command(self, angles, move_time=1000):
        self._validate_angles(angles)
        
        cmd = ""
        for joint, angle in angles.items():
            pulse = int(500 + (2000 * (angle / 180.0)))
            channel = self._get_channel(joint)
            cmd += f"#{channel}P{pulse} "
        cmd += f"T{move_time}\r"
        self.ser.write(cmd.encode())
    
    def emergency_stop(self):
        self.ser.write(b"STOP\r")
    
    def home_position(self):
        self.send_command({
            'base': 90,
            'shoulder': 90,
            'elbow': 90,
            'wrist': 90
        })
    
    def _get_channel(self, joint):
        # Map joint names to channels
        channel_map = {
            'base': 0,
            'shoulder': 1,
            'elbow': 2,
            'wrist': 3,
            'gripper': 4
        }
        return channel_map.get(joint, -1)