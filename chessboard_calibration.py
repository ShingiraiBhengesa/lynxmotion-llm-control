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
    try:
        data = np.load(path)
        return data['mtx'], data['dist']
    except FileNotFoundError:
        print(f"❌ Intrinsic calibration file {path} not found. Run calibrate_camera.py first.")
        return None, None

def generate_object_points():
    """Create 3D points for chessboard corners in mm."""
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM
    return objp

def draw_axes(img, rvec, tvec, camera_matrix, dist_coeffs):
    """Draw 3D axes on the image for pose visualization."""
    axis = np.float32([[0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50]]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 3)  # X-axis (red)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), 3)  # Y-axis (green)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (255, 0, 0), 3)  # Z-axis (blue)
    return img

def main():
    """Capture a chessboard image and compute extrinsic parameters."""
    camera_matrix, dist_coeffs = load_intrinsics()
    if camera_matrix is None or dist_coeffs is None:
        return

    camera = LogitechCamera(resolution=(640, 480))
    print(f"📸 Show a {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} chessboard ({SQUARE_SIZE_MM}mm squares).")
    print("Press SPACE to capture a calibration frame, Q to quit...")

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
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE,
                                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)

    if not ret:
        print("❌ Chessboard not found. Ensure it’s fully visible and well-lit.")
        camera.release()
        cv2.destroyAllWindows()
        return

    print("✅ Chessboard detected. Calibrating...")
    objp = generate_object_points()
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                              criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    
    # Compute extrinsics
    ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    if not ret:
        print("❌ Failed to compute extrinsics.")
        camera.release()
        cv2.destroyAllWindows()
        return

    # Check reprojection error
    projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    error = cv2.norm(corners, projected, cv2.NORM_L2) / len(projected)
    print(f"📏 Reprojection error: {error:.3f} pixels")
    if error > 0.5:
        print("⚠️ Warning: High reprojection error, consider re-capturing with better lighting or angles.")

    # Save calibration
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez(SAVE_PATH, rvec=rvec, tvec=tvec)
    print(f"✅ Calibration saved to {SAVE_PATH}")
    print("Rotation Vector (rvec):\n", rvec)
    print("Translation Vector (tvec):\n", tvec)

    # Preview with axes
    frame_with_corners = cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret)
    frame_with_axes = draw_axes(frame_with_corners, rvec, tvec, camera_matrix, dist_coeffs)
    cv2.imshow('Detected Corners and Axes', frame_with_axes)
    cv2.waitKey(0)

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()