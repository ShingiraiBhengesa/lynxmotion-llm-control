import numpy as np
import cv2

def load_camera_calibration(path='config/camera_pose.npz'):
    data = np.load(path)
    return data['camera_matrix'], data['dist_coeffs'], data['rvec'], data['tvec']

def pixel_to_world_3D(pixel, camera_matrix, dist_coeffs, rvec, tvec, z_world=0):
    """
    Convert a 2D pixel to a 3D world coordinate at a given Z height (e.g., table level).

    Args:
        pixel (tuple): (x, y) pixel coordinates
        camera_matrix, dist_coeffs, rvec, tvec: calibration data
        z_world (float): Z height in mm (e.g., 0 = table)

    Returns:
        tuple: (X, Y, Z) world coordinate in mm
    """
    px = np.array([[pixel]], dtype=np.float32)  # shape: (1, 1, 2)
    undistorted = cv2.undistortPoints(px, camera_matrix, dist_coeffs, P=camera_matrix)

    # Convert rvec to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    R_inv = np.linalg.inv(R)
    tvec = tvec.reshape((3, 1))

    # Convert pixel to normalized camera ray
    uv_homog = np.append(undistorted[0][0], [1.0]).reshape((3, 1))
    cam_ray = np.linalg.inv(camera_matrix) @ uv_homog

    # Solve for scale along ray to intersect Z = z_world
    s = (z_world - R_inv @ tvec)[2, 0] / (R_inv @ cam_ray)[2, 0]
    world_point = R_inv @ (s * cam_ray - tvec)

    return tuple(world_point.flatten())  # (x, y, z)
