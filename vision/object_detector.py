"""Detects colored objects and maps their centers to world coordinates."""

import cv2
import numpy as np
from vision.pixel_to_world_chessboard import load_camera_calibration, pixel_to_world_3D

class ObjectDetector:
    """
    Detects colored objects using HSV filtering and maps centers to 3D world space.
    """

    def __init__(self, debug=False, min_area=800):
        """
        Args:
            debug (bool): If True, display annotated detections.
            min_area (int): Minimum contour area for valid objects.
        """
        self.debug = debug
        self.min_area = min_area
        self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec = load_camera_calibration()
        if any(x is None for x in [self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec]):
            raise RuntimeError("Failed to load camera calibration. Run calibration scripts first.")
        
        # Define HSV color ranges
        self.color_ranges = {
            "red": ((0, 100, 100), (10, 255, 255)),
            "green": ((40, 50, 50), (90, 255, 255)),
            "blue": ((100, 150, 0), (140, 255, 255)),
            "yellow": ((20, 100, 100), (30, 255, 255)),
        }

    def detect_objects(self, frame, z_world=0.0):
        """
        Detects colored objects and computes their centers in pixels and mm.

        Args:
            frame: BGR image from camera.
            z_world (float): Assumed Z height in mm (e.g., table surface).

        Returns:
            List of dicts: {label, confidence, center_px, center_mm, bbox}
        """
        detected_objects = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        world_xyz = pixel_to_world_3D(
                            (cx, cy), self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec, z_world
                        )

                        x, y, w, h = cv2.boundingRect(contour)
                        confidence = min(0.9, area / (self.min_area * 10))

                        detected_objects.append({
                            "label": f"{color}_object",
                            "confidence": confidence,
                            "center_px": (cx, cy),
                            "center_mm": world_xyz,
                            "bbox": (x, y, w, h)
                        })

                        if self.debug:
                            label = f"{color} ({int(world_xyz[0])},{int(world_xyz[1])},{int(world_xyz[2])})"
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                            cv2.putText(frame, label, (cx, cy - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if self.debug:
            cv2.imshow("Detection Debug", frame)
            cv2.waitKey(1)

        return detected_objects