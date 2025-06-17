"""LLM command parser using OpenAI GPT-4 Vision API."""

import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class LLMController:
    """Handles GPT-4 Turbo + Vision-based natural language parsing."""

    def __init__(self):
        self.client = openai.OpenAI()

    def generate_command(self, user_prompt, image_path):
        """Send prompt and image to LLM and return structured robot command.

        Args:
            user_prompt (str): Natural language command from user
            image_path (str): Path to saved image file

        Returns:
            dict: {
                "command": "MOVE" or "GRIP",
                "target": [x, y, z] (if MOVE),
                "gripper": "open"/"close" (if GRIP),
                "error": optional error message
            }
        """
        try:
            with open(image_path, "rb") as img:
                image_bytes = img.read()

            system_msg = (
                "You are a robotics assistant. You receive an image of a workspace "
                "and a user's instruction. Return a JSON command for a 4DOF robotic arm. "
                "Respond ONLY with JSON.\n"
                "Example MOVE: {\"command\": \"MOVE\", \"target\": [120, 150, 50]}\n"
                "Example GRIP: {\"command\": \"GRIP\", \"gripper\": \"open\"}"
            )

            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                max_tokens=300,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_bytes.encode('base64') if hasattr(image_bytes, 'encode') else image_bytes.decode('latin1')}"
                        }}
                    ]}
                ]
            )
        except Exception as e:
            return {"error": str(e)}
