"""LLM command parser using OpenAI GPT-4 Vision API."""

import openai
import os
import json
import base64
from dotenv import load_dotenv

load_dotenv()

class LLMController:
    def __init__(self, model="gpt-4-turbo"):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate_command(self, user_input, image_path):
        try:
            # Encode image to base64
            base64_image = self._encode_image(image_path)
            
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
                                "image_url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        ]
                    }
                ],
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"LLM Error: {str(e)}"}
    
    def _create_system_prompt(self):
        return """You control a Lynxmotion robotic arm. Respond in JSON format:
        {
            "command": "MOVE|GRIP",
            "target": [x,y,z],  // For MOVE only
            "gripper": "open|close"  // For GRIP only
        }"""
    
    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')