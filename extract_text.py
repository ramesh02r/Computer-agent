from __future__ import annotations

import sys
from pathlib import Path as _Path

# from rich.panel import p

_PKG_DIR = _Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import argparse
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from agent import AgentConfig
import pyautogui
from io import BytesIO
from PIL import Image, ImageDraw

from utils import (
    DEFAULT_COMPUTER_USE_MODEL,
    capture_primary_monitor_png,
    collect_text_from_response,
    iter_response_parts,
)

_DESKTOP_SCREEN_URL = "https://desktop.local/screen"
_PAGE_WHEEL_CLICKS = 12
SCROLL_SENSITIVITY = 1.0  
_MAGNITUDE_DIVISOR = 10.0
_POST_SCROLL_CAPTURE_DELAY_S = 0.35
_DESKTOP_EXCLUDED_PREDEFINED: tuple[str, ...] = (
    "open_web_browser",
    "navigate",
    "search",
    "go_back",
    "go_forward",
    "wait_5_seconds",
    "click_at",
    "hover_at",
    "type_text_at",
    "drag_and_drop",
    "key_combination",
)
_SCREEN_DIMS: tuple[int, int] = (0, 0)
_EXTRACTED_BLOCK_BEGIN = "[EXTRACTED_THIS_VIEW]"
_EXTRACTED_BLOCK_END = "[/EXTRACTED_THIS_VIEW]"
_STOP_SCROLLING = "[STOP_SCROLLING]"
OUTPUT_DIR = _PKG_DIR / "extract_text_output_folder"
OUTPUT_FILE = OUTPUT_DIR / "extract_text.txt"


def save_extract_text_result(result: DesktopExtractResult) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(result.as_printable(), encoding="utf-8")
    return OUTPUT_FILE


@dataclass
class DesktopExtractResult:
    """Incremental extracted content + final model text."""

    results: list[str] = field(default_factory=list)
    turns: list[str] = field(default_factory=list)
    final_text: str = ""

    def as_printable(self) -> str:
        lines: list[str] = []
        if self.results:
            lines.append("--- Collected results (all screens) ---")
            for i, c in enumerate(self.results, 1):
                lines.append(f"{i}. {c}")
            lines.append("")
        if self.final_text.strip():
            lines.append("--- Final reply ---")
            lines.append(self.final_text.strip())
        return "\n".join(lines).strip() or "(empty)"

class ExtractText:
    def __init__(self, agent_config: AgentConfig, model: str, query: str):
        self.agent_config = agent_config
        self.model = model
        self.query = query
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.generate_content_config = self.agent_config.get_config()
        self.contents = []

    def run(self):
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        self.png0, self.screen_width, self.screen_height = self._fresh_screen_png()
        self.result = self.extract_visible_text_for_query()
        path = save_extract_text_result(self.result)
        print(f"Saved {path}")
        return self.result

    def _ensure_pyautogui(self):
        pyautogui.FAILSAFE = True
        return pyautogui

    def _fresh_screen_png(self):
        screenshot = pyautogui.screenshot()
        # Optional resize for Retina Macs
        screenshot.save("captured_screen.png")
        buffer = BytesIO()
        resized = screenshot.resize((1000, 1000))
        # Save resized image for debugging
        resized.save("resized_1000.png")
        resized.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        print(f"width: {screenshot.width}, height: {screenshot.height}")
        return png_bytes, screenshot.width, screenshot.height

    def _strip_extracted_blocks(self, text: str) -> str:
        return re.sub(
            re.escape(_EXTRACTED_BLOCK_BEGIN) + r"[\s\S]*?" + re.escape(_EXTRACTED_BLOCK_END),
            "",
            text,
            flags=re.MULTILINE,
        ).strip()

    def extract_visible_text_for_query(
        self,
        max_turns: int = 30,
    ) -> DesktopExtractResult:
        """Observe/act loop: each model turn may request scrolls; each scroll is answered with a fresh
        PNG of the primary monitor, then ``generate_content`` runs again until the model returns
        plain text (task done) or ``max_turns`` is exceeded.

        Parsed ``[EXTRACTED_THIS_VIEW]`` blocks from every turn are merged into ``result.results``.
        Every model text reply is stored in ``result.turns`` and appended to ``self.contents`` for
        the next API call."""
        pg = self._ensure_pyautogui()

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit(
                "Set GEMINI_API_KEY or GEMINI_API_KEY in the environment (e.g. in a .env file)."
            )

        client = self.client
        png0, w0, h0 = self._fresh_screen_png()

        self.contents.append    (
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=png0, mime_type="image/png"),
                    types.Part(
                        text=(
                            f"Screen size: {1000}x{1000} pixels (matches the image).\n\n"
                            f"User request:\n{self.query}\n\n"
                            "Put visible text from each screen in "
                            "[EXTRACTED_THIS_VIEW]...[/EXTRACTED_THIS_VIEW]. "
                            "When the request is satisfied, include [STOP_SCROLLING]."
                        )
                    ),
                ],
            )
        )

        merged_results: list[str] = []
        seen_keys: set[str] = set()

        model_turns: list[str] = []

        def _merge_extracted_from_model_text(text: str) -> None:
            for chunk in self._parse_extracted_block(text):
                key = self._normalize_chunk_key(chunk)
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                merged_results.append(chunk)

        last_text = ""
        for _ in range(max_turns):
            response = client.models.generate_content(
                model=self.model,
                contents=self.contents,
                config=self.generate_content_config,
            )
            print(f"response: {response}")
            if not response.candidates:
                break

            cand = response.candidates[0]
            if cand.content:
                self.contents.append(cand.content)

            last_text = collect_text_from_response(response)
            if last_text.strip():
                model_turns.append(last_text)
            _merge_extracted_from_model_text(last_text)

            if _STOP_SCROLLING in last_text:
                final = self._strip_extracted_blocks(last_text).replace(_STOP_SCROLLING, "").strip()
                return DesktopExtractResult(
                    results=list(merged_results),
                    turns=list(model_turns),
                    final_text=final or last_text.strip(),
                )

            calls = self._extract_function_calls(response)

            if not calls:
                if last_text.strip():
                    final = self._strip_extracted_blocks(last_text).strip()
                    return DesktopExtractResult(
                        results=list(merged_results),
                        turns=list(model_turns),
                        final_text=final or last_text.strip(),
                    )
                break

            # One function_response per tool call; each scroll handler attaches a post-scroll PNG.
            response_parts = [self._execute_function_call(fc) for fc in calls]
            self.contents.append(types.Content(role="user", parts=response_parts))
        
        return DesktopExtractResult(
            results=list(merged_results),
            turns=list(model_turns),
            final_text=last_text.strip(),
        )

    
    def _extract_function_calls(self, response: types.GenerateContentResponse) -> list[types.FunctionCall]:
        out: list[types.FunctionCall] = []
        for part in iter_response_parts(response):
            if part.function_call and part.function_call.name:
                out.append(part.function_call)
        return out





    def _normalize_chunk_key(self, text: str) -> str:
        return " ".join(text.split()).strip().lower()

    def _parse_extracted_block(self, text: str) -> list[str]:
        """Extract content from ``[EXTRACTED_THIS_VIEW] ... [/EXTRACTED_THIS_VIEW]``."""
        if _EXTRACTED_BLOCK_BEGIN not in text or _EXTRACTED_BLOCK_END not in text:
            return []
        pattern = re.compile(
            re.escape(_EXTRACTED_BLOCK_BEGIN) + r"([\s\S]*?)" + re.escape(_EXTRACTED_BLOCK_END),
            re.MULTILINE,
        )
        found: list[str] = []
        for m in pattern.finditer(text):
            block = m.group(1).strip()
            if block:
                found.append(block)
        return found



    def _computer_use_response_fields(self, extra: dict | None = None) -> dict:
        base = {"url": _DESKTOP_SCREEN_URL, "current_url": _DESKTOP_SCREEN_URL}
        if extra:
            base.update(extra)
        return base



    def _capture_screen_after_scroll(self) -> tuple[bytes, int, int]:
        """Always capture a new primary-monitor PNG after scrolling so the model sees current pixels."""
        time.sleep(_POST_SCROLL_CAPTURE_DELAY_S)
        return self._fresh_screen_png()


    def _denorm_xy(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        w, h = _SCREEN_DIMS
        if w <= 0 or h <= 0:
            png, w, h = self._fresh_screen_png()
        x = int(round(float(x_norm) / 1000.0 * w))
        y = int(round(float(y_norm) / 1000.0 * h))
        return max(0, min(w - 1, x)), max(0, min(h - 1, y))


    def _magnitude_to_wheel_clicks(self, magnitude_norm: object) -> int:
        try:
            m = float(magnitude_norm)
        except (TypeError, ValueError):
            m = 800.0
        clicks = int(round(m / _MAGNITUDE_DIVISOR))
        return max(1, min(120, clicks))


    def _pyautogui_wheel_at(self, x: int, y: int, direction: str, clicks: int) -> None:
        pg = pyautogui
        pg.moveTo(x, y, duration=0.1)
        for _ in range(clicks):
            if direction == "up":
                pg.scroll(1)
            elif direction == "down":
                pg.scroll(-1)
            elif direction == "left":
                pg.hscroll(-1)
            elif direction == "right":
                pg.hscroll(1)
            else:
                raise ValueError(direction)
            time.sleep(0.02)
        time.sleep(0.2)


    def _function_response_with_screenshot(
        self,
        fc: types.FunctionCall,
        png_bytes: bytes,
        response_dict: dict,
    ) -> types.Part:
        return types.Part(
            function_response=types.FunctionResponse(
                id=getattr(fc, "id", None),
                name=fc.name,
                response=self._computer_use_response_fields(response_dict),
                parts=[
                    types.FunctionResponsePart.from_bytes(
                        data=png_bytes, mime_type="image/png"
                    )
                ],
            )
        )


    def _handle_scroll_document(self, fc: types.FunctionCall) -> types.Part:
        args = dict(fc.args or {})
        direction = args.get("direction")
        if direction  == "up":
            pyautogui.scroll(7)
        elif direction == "down":
            pyautogui.scroll(-7)
        elif direction == "left":
            pyautogui.hscroll(-7)
        elif direction == "right":
            pyautogui.hscroll(7)
        else:
            raise ValueError(direction)
            
        # --- FIX: Capture new view and return the structured Part response ---
        png, _, _ = self._capture_screen_after_scroll()
        return self._function_response_with_screenshot(
            fc=fc,
            png_bytes=png,
            response_dict={"status": "success", "scrolled": direction}
        )

    

    def _handle_scroll_at(self, fc: types.FunctionCall) -> types.Part:
        args = dict(fc.args or {})
        screen_w, screen_h = pyautogui.size()
        
        # Scale coordinates
        target_x = int((args.get("x", 500) / 1000.0) * screen_w)
        target_y = int((args.get("y", 500) / 1000.0) * screen_h)
        
        pyautogui.moveTo(target_x, target_y)
        
        # --- FIX: Scale down the raw magnitude so it doesn't hyper-scroll ---
        raw_magnitude = args.get('magnitude', 800)
        # Convert 100-800 scale down to 1-5 line increments for PyAutoGUI on Mac
        mac_scroll_clicks = max(1, int(raw_magnitude / 100.0) * 7) 
        
        direction = args.get("direction", "down")
        
        if direction == 'down':
            print(f"Executing scroll_at down: moving {mac_scroll_clicks} units")
            pyautogui.scroll(-mac_scroll_clicks) 
        elif direction == 'up':
            print(f"Executing scroll_at up: moving {mac_scroll_clicks} units")
            pyautogui.scroll(mac_scroll_clicks)
        elif direction in ['left', 'right']:
            clicks = -mac_scroll_clicks if direction == 'right' else mac_scroll_clicks
            pyautogui.hscroll(clicks)

        # Let the UI settle down before capturing
        time.sleep(_POST_SCROLL_CAPTURE_DELAY_S)

        # Capture new view and return response
        png, _, _ = self._capture_screen_after_scroll()
        return self._function_response_with_screenshot(
            fc=fc,
            png_bytes=png,
            response_dict={"status": "success", "scroll_executed": f"{direction} by {mac_scroll_clicks} lines"}
        )


    def _execute_function_call(self, fc: types.FunctionCall) -> types.Part:
        name = fc.name
        if name == "scroll_document":
            print(f"Executing scroll_document function call: {fc}")
            return self._handle_scroll_document(fc)
        if name == "scroll_at":
            print(f"Executing scroll_at function call: {fc}")
            return self._handle_scroll_at(fc)
        png, _, _ = self._fresh_screen_png()
        return self._function_response_with_screenshot(
            fc=fc,
            png_bytes=png,
            response_dict={"error": f"Unsupported tool on desktop extract agent: {name!r}"},
        )