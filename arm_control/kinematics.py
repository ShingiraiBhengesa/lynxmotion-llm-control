"""Inverse kinematics for Lynxmotion robotic arm."""
import math
import os
import yaml

def calculate_ik(x, y, z, grip_angle_d=90.0): # Added grip_angle_d, default to 90.0 for downward gripper
    """Calculate joint angles for target position.
    
    Args:
        x (float): X-coordinate in mm
        y (float): Y-coordinate in mm
        z (float): Z-coordinate in mm
        grip_angle_d (float): Desired gripper pitch angle in degrees relative to horizontal (0=horizontal, 90=down)
        
    Returns:
        dict: Joint angles in degrees
    """
    try:
        # Load dimensions from config
        config_path = os.path.join(os.path.dirname(__file__), '../config/arm_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        BASE_HGT = config['base_height']
        HUMERUS = config['shoulder_length']
        ULNA = config['forearm_length']
        GRIPPER = config['wrist_length'] # Using wrist_length from config as GRIPPER length

        hum_sq = HUMERUS**2
        uln_sq = ULNA**2

        # --- IK Calculations ---

        # Convert grip_angle_d to radians for trigonometric functions
        grip_angle_r = math.radians(grip_angle_d)

        # Base angle (rotation around Z-axis) and radial distance from x,y coordinates
        bas_angle_r = math.atan2(x, y)
        rdist = math.sqrt((x * x) + (y * y))
        
        # 'y_ik' is the radial distance in the 2D arm plane 
        y_ik = rdist

        # Gripper offsets calculated based on desired grip angle and GRIPPER length
        # These offsets are subtracted from the target XYZ to find the wrist joint position
        grip_off_z = math.sin(grip_angle_r) * GRIPPER
        grip_off_y = math.cos(grip_angle_r) * GRIPPER
        
        # Calculate the wrist joint position relative to the shoulder pivot
        # Adjusting target Z by BASE_HGT because Z in IK is usually relative to base pivot plane
        wrist_z = (z - grip_off_z) - BASE_HGT
        wrist_y = y_ik - grip_off_y
        
        # Shoulder to wrist distance squared (s_w_sq) and actual distance (s_w)
        s_w_sq = (wrist_z * wrist_z) + (wrist_y * wrist_y)
        s_w = math.sqrt(s_w_sq)
        
        # Check reachability: If the distance to the wrist target is too long or too short
        if s_w > (HUMERUS + ULNA) or s_w < abs(HUMERUS - ULNA):
            raise ValueError(f"Position ({x}, {y}, {z}) is unreachable by arm geometry. (s_w={s_w:.2f}, Max Reach={HUMERUS+ULNA:.2f}, Min Reach={abs(HUMERUS-ULNA):.2f})")
            
        # Angle from horizontal to the line connecting shoulder and wrist (a1)
        a1 = math.atan2(wrist_z, wrist_y)
        
        # Angle inside the shoulder-elbow-wrist triangle at the shoulder (a2)
        # Using Law of Cosines
        a2 = math.acos((hum_sq - uln_sq + s_w_sq) / (2 * HUMERUS * s_w))
        
        # Shoulder angle relative to the base horizontal plane
        shl_angle_r = a1 + a2
        shl_angle_d = math.degrees(shl_angle_r)
        
        # Elbow angle (angle inside the elbow-shoulder-wrist triangle at the elbow)
        # Using Law of Cosines
        elb_angle_r = math.acos((hum_sq + uln_sq - s_w_sq) / (2 * HUMERUS * ULNA))
        elb_angle_d = math.degrees(elb_angle_r)
        
        # Elbow angle mapping for servo 
        elb_angle_dn = -(180.0 - elb_angle_d) 

        # Wrist angle (calculated based on desired gripper angle, shoulder and elbow)
        wri_angle_d = (grip_angle_d - elb_angle_dn) - shl_angle_d
        
        # Final angles to be returned
        # These are geometric angles, check_joint_limits will enforce physical servo ranges
        base_angle_final = math.degrees(bas_angle_r)
        shoulder_angle_final = shl_angle_d
        elbow_angle_final = elb_angle_dn 
        wrist_angle_final = wri_angle_d

        return {
            'base': base_angle_final,
            'shoulder': shoulder_angle_final,
            'elbow': elbow_angle_final,
            'wrist': wrist_angle_final
        }
        
    except Exception as e:
        print(f"IK Error: {str(e)}")
        return None