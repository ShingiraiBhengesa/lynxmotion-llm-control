"""LLM command parser using OpenAI GPT-4 Vision API."""

import openai
import os
import json
import base64
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class LLMController:
    def __init__(self, model="gpt-4-turbo"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def ask(self, prompt, image=None):
        """
        Query the LLM with a prompt and optional image.

        Args:
            prompt (str): Text prompt for the LLM.
            image (np.ndarray, optional): BGR image array from OpenCV.

        Returns:
            dict: Parsed JSON response from LLM.
        """
        try:
            if image is not None:
                base64_image = self._encode_image(image)
                messages = [
                    {"role": "system", "content": self._create_system_prompt()},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]
            else:
                messages = [
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": prompt}
                ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"LLM Error: {str(e)}"}
    
    def ask_text_only(self, prompt):
        """Fallback method for text-only LLM query."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"LLM Error: {str(e)}"}

    def _create_system_prompt(self):
        return """You control a Lynxmotion robotic arm. The arm operates within a defined workspace.
        X-axis range: -300mm to 300mm
        Y-axis range: 0mm to 400mm
        Z-axis range: 10mm (above table) to 250mm (maximum height).
        
        Respond in JSON format.
        - For MOVE, include a "speed" parameter: "slow", "normal", or "fast".
        
        {
            "command": "MOVE|GRIP",
            "target": [x,y,z],         // For MOVE only
            "speed": "slow|normal|fast", // For MOVE only
            "gripper": "open|close"      // For GRIP only
        }
        Do not generate coordinates outside these bounds. Use only provided object positions.
        """

    def _encode_image(self, image):
        """
        Encode a NumPy image array to base64.

        Args:
            image (np.ndarray): BGR image from OpenCV.

        Returns:
            str: Base64-encoded JPEG image.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a NumPy array")
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')