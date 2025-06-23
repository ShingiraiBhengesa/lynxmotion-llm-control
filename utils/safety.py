"""Safety checks for joint limits and valid 3D workspace."""

import json
import os

def validate_position(x, y, z):
    """Check if position is within workspace."""
    if not (-300 <= x <= 300):
        return False
    if not (0 <= y <= 400):
        return False
    if not (10 <= z <= 250):  # 10mm buffer to avoid table
        return False
    return True

def check_joint_limits(angles):
    """Ensure angles are within safe limits."""
    limits_path = os.path.join(os.path.dirname(__file__), '../arm/joint_limits.json')
    if not os.path.exists(limits_path):
        raise FileNotFoundError(f"Joint limits file {limits_path} not found.")
    limits = json.load(open(limits_path))
    
    for joint, angle in angles.items():
        if joint in limits:
            low, high = limits[joint]
            if not (low <= angle <= high):
                print(f"⚠️ SAFETY: Joint '{joint}' angle {angle:.2f} is outside limits ({low:.2f}, {high:.2f}).")
                return False
    return True