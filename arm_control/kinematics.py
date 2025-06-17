"""Inverse kinematics for 4DOF Lynxmotion robotic arm."""

import math
import yaml
import os

def load_arm_config():
    """Load arm dimensions from YAML config."""
    path = os.path.join(os.path.dirname(__file__), '../config/arm_config.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)['dimensions']

def calculate_ik(x, y, z):
    """Compute joint angles from target 3D position.

    Args:
        x (float): X position in mm
        y (float): Y position in mm
        z (float): Z position in mm

    Returns:
        dict: {
            'base': angle,
            'shoulder': angle,
            'elbow': angle,
            'wrist': angle
        }
    """
    dims = load_arm_config()
    base_height = dims['base_height']
    shoulder_len = dims['shoulder_length']
    forearm_len = dims['forearm_length']
    wrist_len = dims['wrist_length']

    # Convert to planar radius and height
    r = math.sqrt(x**2 + y**2)
    z_offset = z - base_height

    # Base rotation
    base_angle = math.degrees(math.atan2(y, x))

    # Wrist target position
    r_wrist = r - wrist_len
    z_wrist = z_offset

    # Law of cosines for shoulder and elbow
    D = math.sqrt(r_wrist**2 + z_wrist**2)

    cos_elbow = (shoulder_len**2 + forearm_len**2 - D**2) / (2 * shoulder_len * forearm_len)
    cos_elbow = min(1, max(-1, cos_elbow))  # clamp
    elbow_angle_rad = math.acos(cos_elbow)
    elbow_angle = math.degrees(elbow_angle_rad)

    # Shoulder angle
    theta1 = math.atan2(z_wrist, r_wrist)
    theta2 = math.acos((shoulder_len**2 + D**2 - forearm_len**2) / (2 * shoulder_len * D))
    shoulder_angle = math.degrees(theta1 + theta2)

    # Wrist to keep end effector level
    wrist_angle = 180 - (shoulder_angle + elbow_angle)

    return {
        'base': round(base_angle, 2),
        'shoulder': round(shoulder_angle, 2),
        'elbow': round(elbow_angle, 2),
        'wrist': round(wrist_angle, 2)
    }
