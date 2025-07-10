"""Camera calibration tool using a printed chessboard for intrinsic parameters."""

import cv2
import numpy as np
import os
from vision.camera import LogitechCamera

def calibrate_camera(output_path="config/camera_calibration.npz", pattern_size=(9, 6), square_size_mm=30.0):
    """Calibrates the webcam using a chessboard and saves intrinsic parameters (mtx, dist).

    Args:
        output_path (str): Path to save calibration file (.npz).
        pattern_size (tuple): (width, height) of inner chessboard corners (e.g., (9, 6)).
        square_size_mm (float): Size of each chessboard square in mm.
    """
    camera = LogitechCamera(resolution=(640, 480))
    obj_points = []
    img_points = []

    # 3D object points in world space (z=0, in mm)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size_mm

    print(f"📷 Show a {pattern_size[0]}x{pattern_size[1]} chessboard ({square_size_mm}mm squares).")
    print("Press 's' to save frame, ESC to quit. Aim for 15–20 good captures.")
    captured = 0

    while captured < 20:
        ret, frame = camera.capture_frame(undistort=False)
        if not ret:
            print("❌ Failed to capture frame")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
        
        if ret:
            # Refine corners
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                      criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
            cv2.putText(frame, f"Captured: {captured}/15", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Calibration", frame)

            key = cv2.waitKey(500)
            if key == ord('s'):
                img_points.append(corners)
                obj_points.append(objp)
                captured += 1
                print(f"[✔] Saved frame {captured}/20")
            elif key == 27:  # ESC
                break
        else:
            cv2.putText(frame, "Chessboard not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(100) == 27:
                break

    camera.release()
    cv2.destroyAllWindows()

    if captured < 10:
        print(f"⚠️ Only {captured} good captures (need at least 10) — calibration aborted.")
        return

    print("📡 Calibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    
    # Compute reprojection error
    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(obj_points)
    print(f"📏 Reprojection error: {mean_error:.3f} pixels")
    if mean_error > 0.5:
        print("⚠️ Warning: High reprojection error, consider re-capturing images with better angles or lighting.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, mtx=mtx, dist=dist)
    print(f"✅ Calibration saved to {output_path}")
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)

if __name__ == "__main__":
    calibrate_camera()