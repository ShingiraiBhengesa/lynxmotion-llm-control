"""Camera calibration tool using a printed chessboard for intrinsic parameters."""

import cv2
import numpy as np
import os
from vision.camera import LogitechCamera

def calibrate_camera(output_path="config/camera_calibration.npz"):
    """Calibrates the webcam using a 9x6 chessboard and saves intrinsic parameters (mtx, dist).
    
    Args:
        output_path (str): Path to save calibration file (.npz).
    
    Note:
        Run this before chessboard_calibration.py, which computes extrinsics.
    """
    camera = LogitechCamera()
    pattern_size = (9, 6)
    obj_points = []
    img_points = []

    # 3D object points in world space (z=0, unitless)
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    print("📷 Show a printed 9x6 chessboard to the camera. Press 's' to save frame, ESC to quit.")
    captured = 0

    while True:
        ret, frame = camera.capture_frame(undistort=False)
        if not ret:
            print("❌ No frame from camera")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
            cv2.imshow("Calibration", frame)

            key = cv2.waitKey(500)
            if key == ord('s'):
                img_points.append(corners)
                obj_points.append(objp)
                captured += 1
                print(f"[✔] Saved frame {captured}/15")
                if captured >= 15:
                    break
            elif key == 27:  # ESC
                break
        else:
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(100) == 27:
                break

    camera.release()
    cv2.destroyAllWindows()

    if captured > 5:
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
            print("⚠️ Warning: High reprojection error, consider re-capturing images.")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path, mtx=mtx, dist=dist)
        print(f"✅ Calibration saved to {output_path}")
    else:
        print("⚠️ Not enough good captures — calibration aborted.")

if __name__ == "__main__":
    calibrate_camera()