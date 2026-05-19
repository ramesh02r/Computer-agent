

from google import genai
from google.genai import types
from google.genai.types import Content, Part
import time
from dotenv import load_dotenv
import os
import shutil
import pyautogui
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from agent import AgentConfig


load_dotenv()

_PKG_DIR = Path(__file__).resolve().parent
DIR = _PKG_DIR / "find_co_coordinates"



class FindCoordinates:
    def __init__(self, agent_config: AgentConfig, model: str, query: str):
        self.agent_config = agent_config
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.generate_content_config = self.agent_config.get_config()
        self.model = model
        self.query = query
    
    def run(self):
        if DIR.exists():
            shutil.rmtree(DIR)
        DIR.mkdir(parents=True, exist_ok=True)
        self.png0, self.screen_width, self.screen_height = self._fresh_screen_png()
        self.find_coordinates()
    
    def _fresh_screen_png(self):
        screenshot = pyautogui.screenshot()
        screenshot.save(DIR / "captured_screen.png")
        buffer = BytesIO()
        resized = screenshot.resize((1000, 1000))
        # Save resized image for debugging
        resized.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        print(f"width: {screenshot.width}, height: {screenshot.height}")
        return png_bytes, screenshot.width, screenshot.height
    
        
    def find_coordinates(self):
        self.contents=[
            Content(
                role="user",
                parts=[
                    Part.from_bytes(data=self.png0, mime_type="image/png"),
                    Part(text=self.query),
                    Part(text=f"Use click_at funciton call")
                ]
            )
        ]
        response = self.client.models.generate_content(
            model=self.model,
            contents=self.contents,
            config=self.generate_content_config,
        )
        candidate = response.candidates[0]

        print(f"response: {response}")

        part = candidate.content.parts[0]

        function_call = part.function_call

        print(f"function_call: {function_call}")

        x = function_call.args["x"]
        y = function_call.args["y"]


        print(x, y)

        self.mark_screen(x, y)
        print(f"Moving to {x}, {y}")

    def mark_screen(self,x, y):
        x = self.denormalize(x, self.screen_width)
        y = self.denormalize(y, self.screen_height)


        img = Image.open(DIR / "captured_screen.png")
        draw = ImageDraw.Draw(img)
        # Draw red circle
        radius = 20
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline="red",
            width=5
        )

        # Optional crosshair
        draw.line((x - 30, y, x + 30, y), fill="red", width=3)
        draw.line((x, y - 30, x, y + 30), fill="red", width=3)

        # Save result
        marked_path = DIR / "find_co_marked_screen.png"
        img.save(marked_path)
        print(f"Saved {marked_path}")


        screen_w, screen_h = pyautogui.size()
        scale_x  = screen_w / 1000
        scale_y  = screen_h / 1000
        print(f"scale_x: {scale_x}, scale_y: {scale_y}")
        x = int(x * scale_x)
        y = int(y * scale_y)

        return x, y

        

    def denormalize(self, value: int, scale: int) -> int:
        """Convert normalized x coordinate (0-1000) to actual pixel coordinate."""
        return int(value / 1000 * scale)
