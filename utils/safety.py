"""Safety checks for joint limits and valid 3D workspace."""

import json
import os

def validate_position(x, y, z, bounds=None):
    """Check if the target position is within allowed 3D workspace.

    Args:
        x, y, z (float): Cartesian target position in mm
        bounds (dict): Optional manual workspace override

    Returns:
        bool: True if valid
    """
    bounds = bounds or {
        "x": (-250, 250),
        "y": (-250, 250),
        "z": (0, 350)
    }

    return (bounds["x"][0] <= x <= bounds["x"][1] and
            bounds["y"][0] <= y <= bounds["y"][1] and
            bounds["z"][0] <= z <= bounds["z"][1])

def check_joint_limits(joint_angles):
    """Ensure all angles are within joint limits.

    Args:
        joint_angles (dict): joint_name: angle_in_degrees

    Returns:
        bool: True if all angles are valid
    """
    path = os.path.join(os.path.dirname(__file__), '../config/joint_limits.json')
    with open(path) as f:
        limits = json.load(f)

    for joint, angle in joint_angles.items():
        if joint in limits:
            low, high = limits[joint]
            if not (low <= angle <= high):
                print(f"⚠️ {joint} angle {angle}° out of range ({low}–{high})")
                return False
    return True
