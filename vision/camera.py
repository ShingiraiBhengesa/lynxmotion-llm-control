import cv2
import numpy as np
import os

class LogitechCamera:
    """Camera interface for capturing frames from Logitech webcam."""

    def __init__(self, camera_index=0, resolution=(640, 480), calibration_path='config/camera_calibration.npz'):
        """
        Args:
            camera_index (int): Usually 0 for internal or first USB cam.
            resolution (tuple): (width, height)
            calibration_path (str): Path to intrinsic calibration file (.npz).
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam at index {camera_index}. Check connection or index.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.resolution = resolution
        self.calibration_data = self._load_calibration(calibration_path)
        print(f"✅ Camera initialized at {resolution}")

    def _load_calibration(self, path):
        """Load camera intrinsics for undistortion."""
        if os.path.exists(path):
            try:
                data = np.load(path)
                return data['mtx'], data['dist']
            except Exception as e:
                print(f"⚠️ Failed to load calibration from {path}: {e}. Undistortion disabled.")
                return None, None
        print(f"⚠️ Calibration file {path} not found. Run camera calibration to generate it.")
        return None, None

    def capture_frame(self, undistort=True):
        """Capture one frame from the camera.

        Args:
            undistort (bool): Apply lens correction if calibration exists.

        Returns:
            tuple: (success, image frame)
        """
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Failed to capture frame. Check webcam connection.")
            return False, None
        if undistort and self.calibration_data[0] is not None:
            mtx, dist = self.calibration_data
            frame = cv2.undistort(frame, mtx, dist)
        return True, frame

    def release(self):
        """Release the webcam."""
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released")

    def __del__(self):
        self.release()