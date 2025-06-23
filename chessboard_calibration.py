"""Computes camera extrinsics using a single chessboard image and pre-calibrated intrinsics."""

import cv2
import numpy as np
import os
from vision.camera import LogitechCamera

# Chessboard dimensions
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_MM = 30.0  # mm per square
SAVE_PATH = 'config/camera_pose.npz'

def load_intrinsics(path='config/camera_calibration.npz'):
    """Load intrinsic calibration parameters."""
    if os.path.exists(path):
        data = np.load(path)
        return data['mtx'], data['dist']
    raise FileNotFoundError("Intrinsic calibration file not found. Run calibrate_camera.py first.")

def generate_object_points():
    """Create 3D points for chessboard corners in mm."""
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp

def main():
    """Capture a chessboard image and compute extrinsic parameters."""
    camera_matrix, dist_coeffs = load_intrinsics()
    camera = LogitechCamera()
    
    print("📸 Press SPACE to capture a calibration frame, Q to quit...")
    while True:
        ret, frame = camera.capture_frame(undistort=False)
        if not ret:
            print("❌ Failed to capture frame")
            break
        display = frame.copy()
        cv2.putText(display, "Press SPACE to capture, Q to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Calibration View', display)
        key = cv2.waitKey(1)
        if key == ord(' '):
            break
        elif key == ord('q'):
            camera.release()
            cv2.destroyAllWindows()
            return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if not found:
        print("❌ Chessboard not found.")
        camera.release()
        cv2.destroyAllWindows()
        return

    print("✅ Chessboard detected. Calibrating...")
    objp = generate_object_points()
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    
    # Compute extrinsics
    ret, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
    
    # Check reprojection error
    projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    error = cv2.norm(corners2, projected, cv2.NORM_L2) / len(projected)
    print(f"📏 Reprojection error: {error:.3f} pixels")
    if error > 0.5:
        print("⚠️ Warning: High reprojection error, consider re-capturing.")

    # Save calibration
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez(SAVE_PATH, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvec=rvec, tvec=tvec)
    print(f"✅ Calibration saved to {SAVE_PATH}")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
    print("rvec (rotation vector):\n", rvec)
    print("tvec (translation vector):\n", tvec)

    # Preview
    frame_with_corners = cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners2, found)
    cv2.imshow('Detected Corners', frame_with_corners)
    cv2.waitKey(0)
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()