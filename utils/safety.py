"""Safety checks for joint limits and valid 3D workspace."""

import json
import os

def validate_position(x, y, z):
    """Check if position is within workspace"""
    if not (-300 <= x <= 300):
        return False
    if not (0 <= y <= 400):
        return False
    # ADJUSTED: Z-axis minimum to 0 to allow reaching the table surface
    # IMPORTANT: Adjust '0' to a small positive value (e.g., 5 or 10) if Z=0 still crashes physically.
    if not (0 <= z <= 250): 
        return False
    return True

def check_joint_limits(angles):
    """Ensure angles are within safe limits"""
    # Joint limits from the provided AL5D C++ code (in degrees)
    limits = {
        'base': (0.0, 180.0),      # BAS_MIN, BAS_MAX
        'shoulder': (20.0, 140.0), # SHL_MIN, SHL_MAX
        'elbow': (20.0, 165.0),    # ELB_MIN, ELB_MAX
        'wrist': (0.0, 180.0)      # WRI_MIN, WRI_MAX. These are broad, fine-tuning might be needed after testing.
        # Gripper limit is handled separately in main.py
    }
    for joint, angle in angles.items():
        if joint in limits:
            low, high = limits[joint]
            if not (low <= angle <= high):
                print(f"⚠️ SAFETY: Joint '{joint}' angle {angle:.2f} is outside limits ({low:.2f}, {high:.2f}).")
                return False
    return True