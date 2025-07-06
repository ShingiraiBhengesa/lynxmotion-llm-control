"""Inverse kinematics for Lynxmotion robotic arm."""
import math
import os
import yaml
from utils.safety import check_joint_limits

def calculate_ik(x, y, z, grip_angle_d=90.0):
    """Calculate joint angles for target position.
    
    Args:
        x (float): X-coordinate in mm
        y (float): Y-coordinate in mm
        z (float): Z-coordinate in mm
        grip_angle_d (float): Desired gripper pitch angle in degrees (0=horizontal, 90=down)
        
    Returns:
        dict: Joint angles in degrees, or None if unreachable or unsafe
    """
    try:
        # Validate Z to match safety.py
        if z < 10:
            raise ValueError(f"Z-coordinate {z} is below minimum (10mm)")

        # Load dimensions from config
        config_path = os.getenv('ARM_CONFIG_PATH', os.path.join(os.path.dirname(__file__), '../config/arm_config.yaml'))
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        BASE_HGT = config['base_height']
        HUMERUS = config['shoulder_length']
        ULNA = config['forearm_length']
        GRIPPER = config['wrist_length']

        hum_sq = HUMERUS**2
        uln_sq = ULNA**2

        # Convert grip_angle_d to radians
        grip_angle_r = math.radians(grip_angle_d)

        # Base angle and radial distance
        bas_angle_r = math.atan2(x, y)
        rdist = math.sqrt(x * x + y * y)
        if rdist < 1e-6:  # Avoid division by zero
            raise ValueError("Target too close to base (x=0, y=0)")

        y_ik = rdist

        # Gripper offsets
        grip_off_z = math.sin(grip_angle_r) * GRIPPER
        grip_off_y = math.cos(grip_angle_r) * GRIPPER
        
        # Wrist joint position
        wrist_z = z - grip_off_z - BASE_HGT
        wrist_y = y_ik - grip_off_y
        
        # Shoulder to wrist distance
        s_w_sq = wrist_z * wrist_z + wrist_y * wrist_y
        if s_w_sq < 0:
            raise ValueError("Invalid shoulder-to-wrist distance (negative square)")
        s_w = math.sqrt(s_w_sq)
        
        # Check reachability
        max_reach = HUMERUS + ULNA
        min_reach = abs(HUMERUS - ULNA)
        if s_w > max_reach or s_w < min_reach:
            raise ValueError(f"Position ({x:.2f}, {y:.2f}, {z:.2f}) is unreachable. (s_w={s_w:.2f}, Max={max_reach:.2f}, Min={min_reach:.2f})")
            
        # Shoulder angle
        a1 = math.atan2(wrist_z, wrist_y)
        a2_arg = (hum_sq - uln_sq + s_w_sq) / (2 * HUMERUS * s_w)
        a2_arg = max(min(a2_arg, 1.0), -1.0)  # Clip to [-1, 1]
        a2 = math.acos(a2_arg)
        shl_angle_r = a1 + a2
        shl_angle_d = math.degrees(shl_angle_r)
        
        # Elbow angle
        elb_arg = (hum_sq + uln_sq - s_w_sq) / (2 * HUMERUS * ULNA)
        elb_arg = max(min(elb_arg, 1.0), -1.0)  # Clip to [-1, 1]
        elb_angle_r = math.acos(elb_arg)
        elb_angle_d = math.degrees(elb_angle_r)
        elb_angle_dn = -(180.0 - elb_angle_d)

        # Wrist angle
        wri_angle_d = grip_angle_d - elb_angle_dn - shl_angle_d

        # Final angles
        angles = {
            'base': math.degrees(bas_angle_r),
            'shoulder': shl_angle_d,
            'elbow': elb_angle_dn,
            'wrist': wri_angle_d
        }

        # Validate joint limits
        if not check_joint_limits(angles):
            print("❌ IK solution outside joint limits.")
            return None

        print(f"✅ IK solution: {angles}")
        return angles
        
    except Exception as e:
        print(f"❌ IK Error: {str(e)}")
        return None