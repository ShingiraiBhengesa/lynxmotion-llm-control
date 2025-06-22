import cv2
import numpy as np

class ObjectDetector:
    """Detects objects based on color and shape using OpenCV."""

    def __init__(self):
        # Define the lower and upper bounds for the color (e.g., red in HSV)
        # These values will likely need to be tuned for your specific environment and object color.
        # For red, you often need two ranges because red wraps around the HUE circle.
        # Use a tool or experiment to find the precise HSV ranges for your objects.
        # Example for a specific shade of red; adjust as necessary.
        # Example range 1 for red
        self.lower_color1 = np.array([0, 100, 100])
        self.upper_color1 = np.array([10, 255, 255])
        # Example range 2 for red (wraps around hue)
        self.lower_color2 = np.array([170, 100, 100])
        self.upper_color2 = np.array([179, 255, 255])
        
        # If you were detecting green, it might be:
        # self.lower_green = np.array([35, 100, 100])
        # self.upper_green = np.array([85, 255, 255])
        
        # Default area filters for detected contours
        # These also need tuning based on how large your objects appear in the camera feed.
        self.min_area = 500  # Minimum pixel area for a valid object
        self.max_area = 100000 # Maximum pixel area for a valid object

    def detect_objects(self, frame, show=False):
        """
        Detects objects of a specific color using color segmentation and contour analysis.

        Args:
            frame (np.ndarray): BGR image frame.
            show (bool): If True, display the frame with detected objects for debugging.

        Returns:
            List[dict]: A list of detected objects with label, bbox, and center.
                        Example: [{'label': 'red_block', 'confidence': 1.0, 'bbox': (x1, y1, x2, y2), 'center': (cx, cy)}]
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for the specified color range(s)
        mask1 = cv2.inRange(hsv_frame, self.lower_color1, self.upper_color1)
        mask2 = cv2.inRange(hsv_frame, self.lower_color2, self.upper_color2)
        mask = mask1 + mask2 # Combine masks if two ranges are used (for red)

        # Apply morphological operations to clean up the mask
        # These help to remove small noise and close gaps in detected regions
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1) # Reduce noise outside objects
        mask = cv2.dilate(mask, kernel, iterations=2) # Fill small holes in objects

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []

        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area to remove very small (noise) or very large (background) contours
            if self.min_area < area < self.max_area:
                # Get bounding box coordinates
                x, y, w, h = cv2.boundingRect(contour)
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                # Calculate center coordinates
                cx, cy = int(x + w/2), int(y + h/2)

                # For simple colored objects, you can assign a label directly.
                # If you need to distinguish different colored objects, you'd add more
                # color range checks here and assign labels accordingly.
                label = "red_block" # Assuming we are primarily looking for a red block

                obj = {
                    "label": label,
                    "confidence": 1.0, # Assigning a fixed confidence as there's no probabilistic model
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy)
                }
                objects.append(obj)

                if show:
                    # Draw bounding box and label for debugging visualization
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if show:
            cv2.imshow("Detections", frame)
            # You might want a longer waitKey(0) for a static image,
            # or 1 for a video stream to keep it responsive.
            cv2.waitKey(1) 

        return objects