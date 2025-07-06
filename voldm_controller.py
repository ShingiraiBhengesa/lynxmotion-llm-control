import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from vision.pixel_to_world_chessboard import pixel_to_world_3D, load_camera_calibration
import json
import os
import time

class VLMController:
    def __init__(self, model_name="microsoft/Florence-2-base", min_area=500, debug=True):
        """
        Initialize the Vision-Language Model (VLM) controller.
        Args:
            model_name (str): Pretrained model name.
            min_area (int): Minimum area for detected objects.
            debug (bool): Enable debug mode for logging and image saving.
        """
        self.device = "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.min_area = min_area
        self.debug = debug
        
        # Load camera calibration
        self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec = load_camera_calibration()
        if any(x is None for x in [self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec]):
            raise RuntimeError("Failed to load camera calibration. Run calibration scripts first.")
        print(f"✅ VLM Controller Initialized with {model_name} on {self.device}.")

    def detect_objects(self, frame, z_world=0.0):
        """
        Detect objects in the given frame using the Florence-2 model.
        Args:
            frame (np.ndarray): Input image frame in BGR format.
            z_world (float): Z-coordinate in world space for 3D mapping.
        Returns:
            list: List of dictionaries containing object details.
        """
        # Convert frame to RGB for PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Use a detection-specific prompt for Florence-2
        prompt = "Detect colored objects (red, green, blue, yellow) and provide labels and approximate bounding box coordinates if possible."
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        # Generate model output
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=1,
                do_sample=False
            )
        
        # Decode the output
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"Raw output: {generated_text}")  # Debug the raw output

        # Initialize list for detected objects
        detected_objects = []
        
        # Attempt to parse Florence-2's text output (placeholder logic)
        try:
            if "objects:" in generated_text.lower() or "object:" in generated_text.lower():
                # Assume output might list objects with optional coordinates (e.g., "red object at (100,200)")
                text = generated_text.lower()
                if "objects:" in text:
                    object_text = text.split("objects:")[1]
                elif "object:" in text:
                    object_text = text.split("object:")[1]
                else:
                    object_text = text
                
                object_list = object_text.replace("[", "").replace("]", "").split(",")
                for obj in object_list:
                    obj = obj.strip()
                    parts = obj.split(" at ")
                    label = parts[0].strip() if len(parts) > 0 else "unknown"
                    if label not in ["red", "green", "blue", "yellow"]:
                        continue
                    
                    # Approximate coordinates if provided
                    cx, cy = frame.shape[1] / 2, frame.shape[0] / 2  # Default to frame center
                    if len(parts) > 1:
                        coords = parts[1].strip("()").split(",")
                        if len(coords) == 2:
                            try:
                                cx, cy = map(float, coords)
                            except ValueError:
                                print(f"Invalid coordinates for {label}: {coords}")

                    # Calculate bounding box (approximate from center)
                    w, h = 50, 50  # Placeholder width and height; adjust based on output
                    x, y = int(cx - w / 2), int(cy - h / 2)
                    area = w * h
                    if area < self.min_area:
                        continue

                    # Convert to 3D coordinates
                    world_xyz = pixel_to_world_3D(
                        (cx, cy), self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec, z_world
                    )
                    
                    detected_objects.append({
                        "label": f"{label}_object",
                        "confidence": 0.9,  # Placeholder confidence
                        "center_px": (int(cx), int(cy)),
                        "center_mm": world_xyz,
                        "bbox": (int(x), int(y), int(w), int(h))
                    })
            else:
                print("No recognizable object format in output")
        except Exception as e:
            print(f"Error parsing output: {e}")

        # Debug: Save image with detected objects
        if self.debug and detected_objects:
            frame_copy = frame.copy()
            for obj in detected_objects:
                x, y, w, h = obj["bbox"]
                label_text = f"{obj['label']} ({int(obj['center_mm'][0])},{int(obj['center_mm'][1])},{int(obj['center_mm'][2])})"
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (255, 0, 255), 2)
                cv2.putText(frame_copy, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            os.makedirs("debug_images", exist_ok=True)
            cv2.imwrite(f"debug_images/detection_{int(time.time())}.jpg", frame_copy)

        return detected_objects

    def plan_action(self, user_command, frame, detected_objects):
        """
        Plan an action based on user command and detected objects.
        Args:
            user_command (str): User input command.
            frame (np.ndarray): Input image frame.
            detected_objects (list): List of detected object dictionaries.
        Returns:
            dict: JSON-compatible action command.
        """
        objects_seen = [
            {
                "label": obj["label"],
                "position": {
                    "x": round(obj["center_mm"][0], 2),
                    "y": round(obj["center_mm"][1], 2),
                    "z": round(obj["center_mm"][2], 2)
                }
            }
            for obj in detected_objects
        ]
        
        system_prompt = """You control a 5-DOF Lynxmotion robotic arm. The arm operates within:
X-axis: -300mm to 300mm
Y-axis: 0mm to 400mm
Z-axis: 10mm to 250mm

Respond in JSON format:
{
    "command": "MOVE|GRIP",
    "target": [x,y,z],
    "speed": "slow|normal|fast",
    "gripper": "open|close"
}
Use ONLY provided object positions. Return ONE JSON object."""
        
        user_prompt = f"Objects: {json.dumps(objects_seen, indent=2)}\nCommand: {user_command}\nReturn JSON command."
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(
            text=f"{system_prompt}\n{user_prompt}",
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=100,
                num_beams=1,
                do_sample=False
            )
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response from VLM"}

if __name__ == "__main__":
    # Example usage
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        vlm = VLMController(debug=True)
        objects = vlm.detect_objects(frame)
        for obj in objects:
            print(f"Detected: {obj['label']} at {obj['center_mm']} at 03:06 PM CDT, 07/06/2025")
        command = vlm.plan_action("move to red_object", frame, objects)
        print(f"Planned action: {command}")