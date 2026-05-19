"""
Shared helpers for ``screen_agent`` (screenshot, markers, JSON parsing, Gemini retries).

Used by ``vision_sample``, ``next_action``, ``table_rows``, ``vlm``, and ``extract_text``.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
import pyautogui

try:
    import mss
    import mss.tools
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'mss'. Install with: pip install mss"
    ) from e

DEFAULT_COMPUTER_USE_MODEL = "gemini-2.5-computer-use-preview-10-2025"
DEFAULT_CAPTURE_DELAY_S = 10


def load_dotenv_from_repo() -> None:
    """Load ``.env`` from repo root (parent of ``screen_agent/``) then cwd."""
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    load_dotenv()


def wait_before_capture(delay_s: float) -> None:
    if delay_s <= 0:
        return
    secs = int(delay_s)
    print(f"Focus the window to capture. Screenshot in {secs} seconds...")
    time.sleep(delay_s)


def get_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "Set GEMINI_API_KEY or GEMINI_API_KEY in the environment (e.g. in a .env file)."
        )
    return key


def capture_primary_monitor_png() -> tuple[bytes, int, int]:
    """Capture the primary monitor as PNG bytes; return (png_bytes, width, height)."""
    with mss.MSS() as sct:
        mon = sct.monitors[1]
        shot = sct.grab(mon)
        png_bytes = mss.tools.to_png(shot.rgb, shot.size)
        return png_bytes, shot.width, shot.height


def iter_response_parts(
    response: types.GenerateContentResponse,
) -> Iterator[types.Part]:
    if not response.candidates:
        return
    cand = response.candidates[0]
    if not cand.content or not cand.content.parts:
        return
    yield from cand.content.parts


def collect_text_from_response(response: types.GenerateContentResponse) -> str:
    """Concatenate text parts (SDK may omit these in ``response.text`` when mixed with tools)."""
    chunks: list[str] = []
    for part in iter_response_parts(response):
        if part.text:
            chunks.append(part.text)
    if chunks:
        return "\n".join(chunks).strip()
    return (response.text or "").strip()


def extract_json_object(text: str) -> dict:
    """Parse a single JSON object from model output (allows ```json fences)."""
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{{[\s\S]*?\}})\s*```", raw, re.IGNORECASE)
    if fence:
        raw = fence.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end < start:
            raise ValueError("No JSON object found in model response.")
        raw = raw[start : end + 1]
    return json.loads(raw)


def denormalize_xy(x_norm: int, y_norm: int, width: int, height: int) -> tuple[int, int]:
    """Map Gemini Computer Use 0–1000 coordinates to pixel (x, y) on the given image size."""
    if width <= 0 or height <= 0:
        width, height = pyautogui.size()
    x = int(round(x_norm / 1000.0 * width))
    y = int(round(y_norm / 1000.0 * height))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def normalize_xy(x_px: int, y_px: int, width: int, height: int) -> tuple[int, int]:
    """Map pixel (x, y) to 0–1000 coordinates for the given image size."""
    if width <= 0 or height <= 0:
        width, height = pyautogui.size()
    x_norm = int(round(x_px / max(1, width) * 1000.0))
    y_norm = int(round(y_px / max(1, height) * 1000.0))
    return max(0, min(1000, x_norm)), max(0, min(1000, y_norm))


def draw_markers_on_png(
    png_bytes: bytes,
    points: list[dict],
    *,
    radius: int = 18,
) -> bytes:
    """Draw red rings, crosshairs, and labels on a PNG; return new PNG bytes."""
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    r = radius
    for p in points:
        x, y = p["x"], p["y"]
        label = p.get("label") or ""
        cx, cy = max(0, min(im.size[0] - 1, x)), max(0, min(im.size[1] - 1, y))
        color_ring = (255, 0, 0, 255)
        color_line = (255, 80, 0, 255)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color_ring, width=3)
        draw.line((cx - r - 6, cy, cx + r + 6, cy), fill=color_line, width=2)
        draw.line((cx, cy - r - 6, cx, cy + r + 6), fill=color_line, width=2)
        if label:
            tx, ty = cx + r + 4, cy - r - 4
            if tx + 8 > im.size[0]:
                tx = max(0, cx - r - 120)
            if ty < 0:
                ty = cy + r + 4
            draw.rectangle(
                (tx - 2, ty - 2, tx + 6 + len(label) * 8, ty + 18),
                fill=(0, 0, 0, 200),
            )
            draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)

    composed = Image.alpha_composite(im, overlay).convert("RGB")
    buf = io.BytesIO()
    composed.save(buf, format="PNG")
    return buf.getvalue()


def generate_content_with_retries(
    client: genai.Client,
    *,
    model: str,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
    max_attempts: int = 4,
) -> types.GenerateContentResponse:
    """Retry transient 429 / 5xx (common with preview + image payloads)."""
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except genai_errors.APIError as e:
            last_exc = e
            code = int(e.code) if e.code is not None else 0
            if code in (429, 500, 502, 503) and attempt < max_attempts - 1:
                time.sleep(min(12.0, 1.0 * (2**attempt)))
                continue
            raise
    assert last_exc is not None
    raise last_exc
