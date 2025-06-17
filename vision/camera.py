"""Camera interface for capturing frames from Logitech webcam."""

import cv2
import numpy as np
import os

class LogitechCamera:
    """Simple webcam wrapper using OpenCV."""

    def __init__(self, camera_index=0, resolution=(1280, 720)):
        """
        Args:
            camera_index (int): Usually 0 for internal or first USB cam.
            resolution (tuple): (width, height)
        """
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.calibration_data = self._load_calibration()

    def _load_calibration(self):
        """Load camera intrinsics for undistortion."""
        path = os.path.join(os.path.dirname(__file__), '../config/camera_calibration.npz')
        if os.path.exists(path):
            data = np.load(path)
            return data['mtx'], data['dist']
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
            return False, None

        if undistort and self.calibration_data[0] is not None:
            mtx, dist = self.calibration_data
            frame = cv2.undistort(frame, mtx, dist)

        return True, frame

    def release(self):
        """Release the webcam."""
        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.release()
