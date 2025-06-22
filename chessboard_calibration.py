"This script captures a chessboard image, detects corners, and calibrates the camera."
import cv2
import numpy as np
import os

# Chessboard dimensions
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_MM = 30.0  # mm per square

SAVE_PATH = 'config/camera_pose.npz'

def generate_object_points():
    """Create 3D points for the chessboard corners in mm."""
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp

def main():
    cap = cv2.VideoCapture(0)
    print("📸 Press SPACE to capture a calibration frame...")
    while True:
        ret, frame = cap.read()
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
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if not found:
        print("❌ Chessboard not found.")
        return

    print("✅ Chessboard detected. Calibrating...")

    objp = generate_object_points()
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # Intrinsic calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [objp], [corners2], gray.shape[::-1], None, None)

    rvec = rvecs[0]
    tvec = tvecs[0]

    # Save calibration
    os.makedirs('config', exist_ok=True)
    np.savez(SAVE_PATH,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rvec=rvec,
             tvec=tvec)

    print(f"✅ Calibration saved to {SAVE_PATH}")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
    print("rvec (rotation vector):\n", rvec)
    print("tvec (translation vector):\n", tvec)

    # Optional preview
    frame_with_corners = cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners2, found)
    cv2.imshow('Detected Corners', frame_with_corners)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
