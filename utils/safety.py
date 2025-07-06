"""Safety checks for joint limits and valid 3D workspace."""

import json
import os
import yaml

def validate_position(x, y, z):
    """Check if position is within workspace.

    Args:
        x (float): X-coordinate in mm.
        y (float): Y-coordinate in mm.
        z (float): Z-coordinate in mm.

    Returns:
        bool: True if position is safe, False otherwise.
    """
    try:
        x, y, z = float(x), float(y), float(z)
    except (TypeError, ValueError):
        print(f"⚠️ SAFETY: Invalid position coordinates ({x}, {y}, {z})")
        return False

    # Load arm dimensions for workspace limits
    config_path = os.getenv('ARM_CONFIG_PATH', os.path.join(os.path.dirname(__file__), '../config/arm_config.yaml'))
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        max_reach = config['shoulder_length'] + config['forearm_length'] + config['wrist_length']
        min_reach = abs(config['shoulder_length'] - config['forearm_length'])
        base_height = config['base_height']
    except (FileNotFoundError, KeyError) as e:
        print(f"⚠️ SAFETY: Failed to load arm_config.yaml: {e}. Using default workspace limits.")
        max_reach, min_reach, base_height = 300.0, 50.0, 50.0

    # Workspace bounds
    rdist = (x**2 + y**2)**0.5
    if not (-max_reach <= x <= max_reach):
        print(f"⚠️ SAFETY: X-coordinate {x:.2f} outside limits [-{max_reach:.2f}, {max_reach:.2f}]")
        return False
    if not (min_reach <= rdist <= max_reach):
        print(f"⚠️ SAFETY: Radial distance {rdist:.2f} outside limits [{min_reach:.2f}, {max_reach:.2f}]")
        return False
    if not (10 <= z <= base_height + max_reach):
        print(f"⚠️ SAFETY: Z-coordinate {z:.2f} outside limits [10, {base_height + max_reach:.2f}]")
        return False
    return True

def check_joint_limits(angles):
    """Ensure angles are within safe limits.

    Args:
        angles (dict): Joint angles in degrees (e.g., {'base': 90, 'shoulder': 90, ...}).

    Returns:
        bool: True if angles are safe, False otherwise.
    """
    limits_path = os.path.join(os.path.dirname(__file__), '../config/joint_limits.json')
    try:
        with open(limits_path) as f:
            limits = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️ SAFETY: Failed to load joint limits: {e}")
        return False

    if not isinstance(angles, dict):
        print(f"⚠️ SAFETY: Angles must be a dictionary, got {type(angles)}")
        return False

    for joint, angle in angles.items():
        try:
            angle = float(angle)
            if joint in limits:
                low, high = limits[joint]
                if not (low <= angle <= high):
                    print(f"⚠️ SAFETY: Joint '{joint}' angle {angle:.2f} outside limits ({low:.2f}, {high:.2f})")
                    return False
            else:
                print(f"⚠️ SAFETY: Joint '{joint}' not found in limits")
                return False
        except (TypeError, ValueError):
            print(f"⚠️ SAFETY: Invalid angle for joint '{joint}': {angle}")
            return False
    return True