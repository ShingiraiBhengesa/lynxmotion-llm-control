"""Inverse kinematics for Lynxmotion robotic arm."""
import math
import os
import yaml

def calculate_ik(x, y, z):
    """Calculate joint angles for target position.
    
    Args:
        x (float): X-coordinate in mm
        y (float): Y-coordinate in mm
        z (float): Z-coordinate in mm
        
    Returns:
        dict: Joint angles in degrees
    """
    try:
        # Load dimensions from config
        config_path = os.path.join(os.path.dirname(__file__), '../config/arm_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        L1 = config['base_height']
        L2 = config['shoulder_length']
        L3 = config['forearm_length']
        
        # Base rotation
        base_angle = math.degrees(math.atan2(y, x))
        r = math.sqrt(x**2 + y**2)
        h = z - L1
        D = math.sqrt(r**2 + h**2)
        
        # Check reachability
        if D > (L2 + L3) or D < abs(L2 - L3):
            raise ValueError(f"Position ({x}, {y}, {z}) is unreachable")
            
        # Shoulder angle
        shoulder_angle = math.acos((L2**2 + D**2 - L3**2) / (2 * L2 * D))
        shoulder_angle += math.atan2(h, r)
        shoulder_angle = math.degrees(shoulder_angle)
        
        # Elbow angle
        elbow_angle = math.acos((L2**2 + L3**2 - D**2) / (2 * L2 * L3))
        elbow_angle = math.degrees(elbow_angle) - 180
        
        # Wrist angle
        wrist_angle = -shoulder_angle - elbow_angle
        
        return {
            'base': base_angle,
            'shoulder': shoulder_angle,
            'elbow': elbow_angle,
            'wrist': wrist_angle
        }
        
    except Exception as e:
        print(f"IK Error: {str(e)}")
        return None