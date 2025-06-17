"""YOLOv8 object detection using Ultralytics module."""

from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    """Wraps YOLOv8 inference using pre-trained or custom model."""

    def __init__(self, model_path='yolov8n.pt', confidence=0.4):
        """
        Args:
            model_path (str): Path to YOLOv8 model (.pt)
            confidence (float): Minimum confidence to accept detection
        """
        self.model = YOLO(model_path)
        self.conf = confidence

    def detect_objects(self, frame, show=False):
        """Run detection on a frame and return results.

        Args:
            frame (np.ndarray): BGR image
            show (bool): If True, show image with boxes (for debug)

        Returns:
            List[dict]: [{
                'label': 'bottle',
                'confidence': 0.87,
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy)
            }]
        """
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        objects = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                obj = {
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy)
                }
                objects.append(obj)

                if show:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if show:
            cv2.imshow("Detections", frame)
            cv2.waitKey(1)

        return objects
