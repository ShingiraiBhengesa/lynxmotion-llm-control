"""LLM command parser using OpenAI GPT-4 Vision API."""

import openai
import base64
import os
import json
import cv2
from dotenv import load_dotenv

load_dotenv()

class LLMController:
    def __init__(self, model="gpt-4-vision-preview"):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate_command(self, user_input, image):
        try:
            base64_image = self._encode_image(image)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._create_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_input},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            return {"error": f"LLM Error: {str(e)}"}

    def _create_system_prompt(self):
        return """You control a Lynxmotion robotic arm. The arm operates within a defined workspace.
X-axis range: -300mm to 300mm
Y-axis range: 0mm to 400mm
Z-axis range: 0mm (table surface) to 250mm (maximum height).

Respond in JSON format with commands and target coordinates ONLY within these valid ranges:
{
    "command": "MOVE|GRIP",
    "target": [x,y,z],  // For MOVE only, must be within workspace limits
    "gripper": "open|close"  // For GRIP only
}
Do not generate coordinates outside these bounds. Be precise with the target coordinates.
"""

    def _encode_image(self, image):
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer).decode("utf-8")
