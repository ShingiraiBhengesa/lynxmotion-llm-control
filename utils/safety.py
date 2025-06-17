"""Safety checks for joint limits and valid 3D workspace."""

import json
import os

def validate_position(x, y, z):
    """Check if position is within workspace"""
    if not (-300 <= x <= 300):
        return False
    if not (0 <= y <= 400):
        return False
    if not (0 <= z <= 250):
        return False
    return True

def check_joint_limits(angles):
    """Ensure angles are within safe limits"""
    limits = {
        'base': (0, 180),
        'shoulder': (30, 120),
        'elbow': (0, 180),
        'wrist': (0, 180)
    }
    for joint, angle in angles.items():
        if joint in limits:
            low, high = limits[joint]
            if not (low <= angle <= high):
                return False
    return True