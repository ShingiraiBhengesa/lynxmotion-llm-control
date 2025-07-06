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
        self.device = "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device).half()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.min_area = min_area
        self.debug = debug
        self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec = load_camera_calibration()
        if any(x is None for x in [self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec]):
            raise RuntimeError("Failed to load camera calibration. Run calibration scripts first.")
        print(f"✅ VLM Controller Initialized with {model_name} on {self.device}.")

    def detect_objects(self, frame, z_world=0.0):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prompt = "Detect colored objects (red, green, blue, yellow) and provide bounding boxes and labels."
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=1,
                do_sample=False
            )
        results = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        detections = self.processor.post_process_object_detection(outputs, image.size)

        detected_objects = []
        for det in detections:
            label = det["labels"][0] if det["labels"] else "unknown"
            if label not in ["red", "green", "blue", "yellow"]:
                continue
            box = det["boxes"][0]
            score = det["scores"][0] if det["scores"] else 0.9
            
            x, y, w, h = box
            area = w * h
            if area < self.min_area:
                continue
            cx, cy = x + w / 2, y + h / 2
            
            world_xyz = pixel_to_world_3D(
                (cx, cy), self.camera_matrix, self.dist_coeffs, self.rvec, self.tvec, z_world
            )
            
            detected_objects.append({
                "label": f"{label}_object",
                "confidence": float(score),
                "center_px": (int(cx), int(cy)),
                "center_mm": world_xyz,
                "bbox": (int(x), int(y), int(w), int(h))
            })

            if self.debug:
                frame_copy = frame.copy()
                label_text = f"{label} ({int(world_xyz[0])},{int(world_xyz[1])},{int(world_xyz[2])})"
                cv2.rectangle(frame_copy, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 255), 2)
                cv2.putText(frame_copy, label_text, (int(cx), int(cy - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                os.makedirs("debug_images", exist_ok=True)
                cv2.imwrite(f"debug_images/detection_{int(time.time())}.jpg", frame_copy)

        return detected_objects

    def plan_action(self, user_command, frame, detected_objects):
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