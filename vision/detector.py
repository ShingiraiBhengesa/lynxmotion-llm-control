import cv2
import numpy as np

class ObjectDetector:
    """Detects objects based on color, or shape/brightness for 'colorless' objects, using OpenCV."""

    def __init__(self):
        # Define HSV color ranges for various objects.
        # These ranges are initial estimates and will require fine-tuning
        # based on your specific lighting conditions and object colors.
        # Use an HSV color picker tool (many online or simple OpenCV scripts)
        # to find the precise ranges for your environment.

        # Red: often requires two ranges because Hue wraps around 0/179
        self.color_ranges = {
            "red_object": [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([179, 255, 255]))
            ],
            # Green: Hue typically around 40-80
            "green_object": [
                (np.array([40, 40, 40]), np.array([80, 255, 255]))
            ],
            # Blue: Hue typically around 100-140
            "blue_object": [
                (np.array([100, 100, 100]), np.array([140, 255, 255]))
            ],
            # Yellow: Hue typically around 20-35
            "yellow_object": [
                (np.array([20, 100, 100]), np.array([35, 255, 255]))
            ],
            # Black: Low Value (brightness), low Saturation
            # Note: Black detection can be tricky and might pick up shadows.
            "black_object": [
                (np.array([0, 0, 0]), np.array([179, 50, 50])) # Low V (0-50), Low S (0-50)
            ],
            # White/Colorless: High Value (brightness), low Saturation
            # Note: This might pick up highlights or bright backgrounds.
            "white_object": [
                (np.array([0, 0, 200]), np.array([179, 50, 255])) # High V (200-255), Low S (0-50)
            ]
        }
        
        # General parameters for contour filtering (adjust based on object sizes)
        self.min_object_area = 500  # Minimum pixel area for a detected object
        self.max_object_area = 100000 # Maximum pixel area for a detected object (to avoid detecting large background elements)

        # Parameters for cup detection (might be larger, less defined color)
        # These are for the "colorless" cup based on its general appearance/shape
        self.cup_min_area = 5000  # Cups are typically larger
        self.cup_max_area = 250000 
        self.cup_aspect_ratio_range = (0.5, 1.5) # Example for a circular/elliptical top view
        self.cup_min_vertices = 6 # For approximating a circle/ellipse

    def detect_objects(self, frame, show=False):
        """
        Detects various objects based on color, and 'colorless' cups based on brightness/shape.

        Args:
            frame (np.ndarray): BGR image frame.
            show (bool): If True, display the frame with detected objects for debugging.

        Returns:
            List[dict]: A list of detected objects.
                        Example: [{'label': 'red_object', 'confidence': 1.0, 'bbox': (x1, y1, x2, y2), 'center': (cx, cy)}]
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Use a copy of the frame for drawing, so the original isn't modified if not showing
        display_frame = frame.copy() if show else None

        detected_objects = []

        # --- Detect Colored Objects ---
        for label, ranges in self.color_ranges.items():
            combined_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
            for (lower, upper) in ranges:
                mask = cv2.inRange(hsv_frame, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Apply morphological operations to clean up the mask
            kernel = np.ones((5,5), np.uint8)
            mask_processed = cv2.erode(combined_mask, kernel, iterations=1)
            mask_processed = cv2.dilate(mask_processed, kernel, iterations=2)
            
            # Find contours in the processed mask
            contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                # Filter by area to remove noise and overly large regions
                if self.min_object_area < area < self.max_object_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx, cy = int(x + w/2), int(y + h/2)

                    obj = {
                        "label": label,
                        "confidence": 1.0, # Fixed confidence for rule-based detection
                        "bbox": (x, y, x + w, y + h),
                        "center": (cx, cy)
                    }
                    detected_objects.append(obj)

                    if show and display_frame is not None:
                        # Define color for drawing based on object label
                        draw_color = (0, 255, 0) # Default green
                        if "red" in label: draw_color = (0, 0, 255)
                        elif "blue" in label: draw_color = (255, 0, 0)
                        elif "yellow" in label: draw_color = (0, 255, 255)
                        elif "black" in label: draw_color = (100, 100, 100) # Dark gray
                        elif "white" in label: draw_color = (255, 255, 255) # White

                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), draw_color, 2)
                        cv2.putText(display_frame, f"{label}",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

        # --- Detect Colorless/White Cups (based on brightness/shape) ---
        # This part assumes a white/colorless cup might be brighter than its surroundings
        # and has a somewhat circular/elliptical opening or specific rectangular body.
        
        # Convert to grayscale for general intensity analysis
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold for bright objects (potential white cups) on a darker background
        # You might need to experiment with THRESH_BINARY_INV if cups are dark on a light background.
        _, cup_mask = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY) # Adjust 200 based on cup brightness

        # Clean up the cup mask
        kernel_cup = np.ones((7,7), np.uint8) # Larger kernel for larger objects
        cup_mask = cv2.erode(cup_mask, kernel_cup, iterations=1)
        cup_mask = cv2.dilate(cup_mask, kernel_cup, iterations=2)

        contours_cup, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_cup:
            area = cv2.contourArea(contour)
            if self.cup_min_area < area < self.cup_max_area:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True) # Approximate contour with polygon

                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = int(x + w/2), int(y + h/2)
                aspect_ratio = float(w) / h

                # Basic shape check for a cup (e.g., roughly circular/oval top or rectangular body)
                # This is a very simplified check and might need more sophistication.
                # Example for a cylindrical cup viewed from top (circular) or side (rectangular)
                is_cup_shape = False
                if len(approx) >= self.cup_min_vertices and (self.cup_aspect_ratio_range[0] < aspect_ratio < self.cup_aspect_ratio_range[1]):
                    is_cup_shape = True # It's somewhat circular/elliptical
                elif 0.2 < aspect_ratio < 0.8: # Could be a tall, narrow rectangular body from side view
                    is_cup_shape = True

                if is_cup_shape:
                    obj = {
                        "label": "cup",
                        "confidence": 1.0,
                        "bbox": (x, y, x + w, y + h),
                        "center": (cx, cy)
                    }
                    detected_objects.append(obj)

                    if show and display_frame is not None:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 2) # Cyan for cup
                        cv2.putText(display_frame, "cup", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if show and display_frame is not None:
            cv2.imshow("Object Detections", display_frame)
            cv2.waitKey(1)

        return detected_objects