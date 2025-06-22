import cv2
import numpy as np
from vision.pixel_to_world_chessboard import (
    load_camera_calibration,
    pixel_to_world_3D
)

class ObjectDetector:
    """
    ObjectDetector detects colored objects in the camera frame using HSV filtering.
    It also maps the center of each object from image space (pixels) to world space (mm)
    using full 3D transformation based on camera extrinsic calibration.
    """

    def __init__(self, debug=False):
        """
        Initializes color ranges and loads camera calibration data.
        :param debug: If True, display annotated detections in a debug window.
        """
        self.debug = debug
        self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec = load_camera_calibration()
        self.min_area = 800  # Minimum area to qualify as a valid object

        # Define HSV color ranges for detection
        self.color_ranges = {
            "red": ((0, 100, 100), (10, 255, 255)),
            "green": ((40, 50, 50), (90, 255, 255)),
            "blue": ((100, 150, 0), (140, 255, 255)),
            "yellow": ((20, 100, 100), (30, 255, 255)),
        }

    def detect_objects(self, frame):
        """
        Detects all color-based objects in the frame and computes both pixel and real-world center.
        :param frame: BGR image from the camera
        :return: List of dictionaries containing object info: label, center_px, center_mm, bbox
        """
        detected_objects = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color, (lower, upper) in self.color_ranges.items():
            # Create mask and clean it with morphological operations
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # Detect contours on the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        # Compute image center
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # Project to world coordinates
                        world_xyz = pixel_to_world_3D(
                            (cx, cy),
                            self.camera_matrix,
                            self.dist_coeffs,
                            self.rvec,
                            self.tvec,
                            z_world=0.0  # assume objects are on table surface
                        )

                        # Bounding box for visualization or tracking
                        x, y, w, h = cv2.boundingRect(contour)

                        detected_objects.append({
                            "label": f"{color}_object",
                            "confidence": 0.9,
                            "center_px": (cx, cy),
                            "center_mm": world_xyz,
                            "bbox": (x, y, w, h)
                        })

                        # Draw overlays in debug mode
                        if self.debug:
                            label = f"{color} ({int(world_xyz[0])},{int(world_xyz[1])},{int(world_xyz[2])})"
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                            cv2.putText(frame, label, (cx, cy - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if self.debug:
            cv2.imshow("Detection Debug", frame)
            cv2.waitKey(1)

        return detected_objects
