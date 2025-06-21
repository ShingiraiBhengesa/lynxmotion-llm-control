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
    limits = {
        'base': (0, 180),
        'shoulder': (30, 120),
        'elbow': (0, 180),
        # ADJUSTED: Wrist joint limits to prevent self-collision
        # YOU MUST CHANGE (25, 180) to values that prevent your specific crash based on physical testing.
        # Example: If crashing below 25 degrees, use (25, 180). If crashing above 160 degrees, use (0, 160).
        'wrist': (25, 180) # Placeholder: Update this based on your physical observation and testing.
    }
    for joint, angle in angles.items():
        if joint in limits:
            low, high = limits[joint]
            if not (low <= angle <= high):
                print(f"DEBUG: Joint '{joint}' angle {angle} outside limits ({low}, {high})") 
                return False
    return True