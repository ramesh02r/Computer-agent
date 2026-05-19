"""
Find matching table rows on the live screen (no table file).

Same structure as ``find_co.py``:
  agent = TableRows(agent_config, model=..., query=...)
  result = agent.run()
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


import pyautogui
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import Content, Part
from PIL import Image, ImageDraw

from agent import AgentConfig
from system_prompts import TABLE_COLUMN_DISCOVERY_SUFFIX, TABLE_ROW_MATCHING_SUFFIX
from utils import (
    DEFAULT_COMPUTER_USE_MODEL,
    collect_text_from_response,
    extract_json_object,
    iter_response_parts,
)

load_dotenv()

DEFAULT_TABLE_MODEL = DEFAULT_COMPUTER_USE_MODEL
DEFAULT_OUT_DIR = _PKG_DIR / "table_rows_output"
_MODEL_W = 1000
_MODEL_H = 1000
_PAGE_WHEEL_CLICKS = int(os.environ.get("TABLE_PAGE_WHEEL_CLICKS", "8"))
_POST_SCROLL_DELAY_S = 0.35

TABLE_ROWS_CU_EXCLUDED: tuple[str, ...] = (
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
    "scroll_document",
    "scroll_at",
)

_ROW_COLORS_RGBA: tuple[tuple[int, int, int, int], ...] = (
    (255, 60, 60, 90),
    (60, 180, 255, 90),
    (80, 220, 120, 90),
    (255, 180, 40, 90),
    (200, 100, 255, 90),
)


@dataclass
class TablePassResult:
    pass_index: int
    screenshot_png: bytes
    screen_width: int
    screen_height: int
    analysis: dict
    rows_on_pass: list[dict]
    marked_png: bytes | None = None


@dataclass
class TableRowsResult:
    query: str
    matching_rows: list[dict] = field(default_factory=list)
    passes: list[TablePassResult] = field(default_factory=list)
    column_discovery_screenshot: bytes | None = None
    column_discovery: dict | None = None
    stop_reason: str = ""
    saved_paths: dict[str, Path] = field(default_factory=dict)


class TableRows:
    def __init__(self, agent_config: AgentConfig, model: str, query: str):
        self.agent_config = agent_config
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.generate_content_config = self.agent_config.get_config()
        self.model = model
        self.query = query.strip()
        self.screen_width = 0
        self.screen_height = 0
        self.table_context: dict = {}
        self.seen_keys: set[str] = set()
        self.max_scroll_passes = 8
        self._scroll_direction: str = "down"
        self._last_capture_hash: str | None = None

    def run(self, *, max_scroll_passes: int = 8) -> TableRowsResult:
        self.max_scroll_passes = max_scroll_passes
        pyautogui.FAILSAFE = True
        if DEFAULT_OUT_DIR.exists():
            shutil.rmtree(DEFAULT_OUT_DIR)
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        result = TableRowsResult(query=self.query)

        print(f"Query: {self.query}")
        print("--- Column discovery (no scroll) ---")
        discovery_full_png = self._discover_columns()

        num = self.table_context.get("filter_column_number")
        name = self.table_context.get("filter_column_name")
        result.column_discovery_screenshot = discovery_full_png
        ctx_save = {k: v for k, v in self.table_context.items() if k != "discovery_png"}
        result.column_discovery = {
            "analysis": self.table_context.get("discovery_analysis"),
            "table_context": ctx_save,
        }
        if num is not None:
            print(f"  Locked filter column: #{num} ({name})")
        else:
            print("  Warning: could not lock filter column.")

        consecutive_empty = 0
        for pass_index in range(self.max_scroll_passes):
            print(f"--- Row pass {pass_index + 1}/{self.max_scroll_passes} ---")
            model_png, full_png, w, h = self._fresh_screen_png()
            capture_hash = self._png_hash(full_png)
            if pass_index > 0 and self._last_capture_hash == capture_hash:
                result.stop_reason = (
                    "Duplicate screenshot after scroll — view did not change."
                )
                print(f"Stopping: {result.stop_reason}")
                break
            analysis = self._match_rows_on_screen(model_png, pass_index)
            rows_on_pass = self._collect_rows(analysis, w, h, pass_index, result)

            print(
                f"  New: {len(rows_on_pass)} | Total: {len(result.matching_rows)}"
            )
            for row in rows_on_pass:
                print(f"    • {row.get('cells_summary') or row.get('row_key')}")

            marked_png = None
            if rows_on_pass:
                consecutive_empty = 0
                self._update_scroll_anchor(rows_on_pass, h)
                marked_png = self._mark_rows_on_png(full_png, rows_on_pass)
            else:
                consecutive_empty += 1

            result.passes.append(
                TablePassResult(
                    pass_index=pass_index,
                    screenshot_png=full_png,
                    screen_width=w,
                    screen_height=h,
                    analysis=analysis,
                    rows_on_pass=rows_on_pass,
                    marked_png=marked_png,
                )
            )

            if not self._should_scroll(analysis, consecutive_empty):
                result.stop_reason = analysis.get("notes") or "No further scrolling."
                print(f"Stopping: {result.stop_reason}")
                break
            if pass_index >= self.max_scroll_passes - 1:
                result.stop_reason = f"Reached max passes ({self.max_scroll_passes})."
                print(f"Stopping: {result.stop_reason}")
                break
            if not self._scroll_with_verify(full_png):
                result.stop_reason = (
                    analysis.get("notes")
                    or "Screen unchanged after scroll; stopping to avoid duplicate passes."
                )
                print(f"Stopping: {result.stop_reason}")
                break
        else:
            result.stop_reason = f"Completed {self.max_scroll_passes} passes."

        return result

    def _fresh_screen_png(self) -> tuple[bytes, bytes, int, int]:
        """Capture screen; return (model_1000_png, full_resolution_png, width, height)."""
        screenshot = pyautogui.screenshot()
        w, h = screenshot.width, screenshot.height
        self.screen_width, self.screen_height = w, h
        screenshot.save(DEFAULT_OUT_DIR / "captured_screen.png")

        full_buf = BytesIO()
        screenshot.save(full_buf, format="PNG")
        full_png = full_buf.getvalue()

        model_buf = BytesIO()
        resized = screenshot.resize((_MODEL_W, _MODEL_H))
        resized.save(DEFAULT_OUT_DIR / "resized_1000.png")
        resized.save(model_buf, format="PNG")
        model_png = model_buf.getvalue()

        print(f"physical: {w}x{h}, model: {_MODEL_W}x{_MODEL_H}")
        return model_png, full_png, w, h

    def _discover_columns(self) -> bytes:
        model_png, full_png, w, h = self._fresh_screen_png()
        text = self._ask_model(
            model_png,
            user_text=(
                f"Screen: {_MODEL_W}x{_MODEL_H} pixels.\n"
                "No scrolling. Read headers and return filter_column_number.\n"
                f"User query:\n{self.query}"
            ),
            extra_system=TABLE_COLUMN_DISCOVERY_SUFFIX,
        )
        analysis = self._parse_json(text)
        cols = analysis.get("columns_visible") or []
        col_strs = [str(c) for c in cols] if isinstance(cols, list) else []
        self.table_context = self._build_table_context(col_strs, analysis)
        if col_strs:
            print(f"  Columns: {col_strs}")
        return full_png

    def _match_rows_on_screen(self, png: bytes, pass_index: int) -> dict:
        num = self.table_context.get("filter_column_number")
        name = self.table_context.get("filter_column_name")
        spec = self.table_context.get("filter_spec_text", "")
        found = ""
        if self.seen_keys:
            found = "\nAlready recorded: " + ", ".join(sorted(self.seen_keys)[:40])
        scroll_note = ""
        if pass_index > 0:
            scroll_note = f"\nScrolled view (pass {pass_index + 1}). Use column #{num}."
        text = self._ask_model(
            png,
            user_text=(
                f"Screen: {_MODEL_W}x{_MODEL_H} pixels.\n"
                f"Row pass {pass_index + 1}. Filter column #{num} ({name}).\n"
                f"{spec}\n"
                f"Query:\n{self.query}"
                + scroll_note
                + found
            ),
            extra_system=TABLE_ROW_MATCHING_SUFFIX,
        )
        return self._parse_json(text)

    def _ask_model(self, png: bytes, *, user_text: str, extra_system: str) -> str:
        contents = [
            Content(
                role="user",
                parts=[
                    Part.from_bytes(data=png, mime_type="image/png"),
                    Part(text=user_text),
                ],
            )
        ]
        config = types.GenerateContentConfig(
            system_instruction=(
                self.agent_config.system_prompt("4") + "\n" + extra_system
            ),
            temperature=0.1,
            max_output_tokens=8192,
            tools=self.generate_content_config.tools,
            tool_config=getattr(self.generate_content_config, "tool_config", None),
        )
        for attempt in range(2):
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            print(f"response: {response}")
            text = (response.text or "").strip() or collect_text_from_response(
                response
            ).strip()
            if text:
                return text

            tool_names: list[str] = []
            for part in iter_response_parts(response):
                if part.function_call and part.function_call.name:
                    tool_names.append(part.function_call.name)
            if tool_names and attempt == 0:
                print(
                    f"  Model returned tool call(s) {tool_names}; "
                    "retrying for JSON-only reply..."
                )
                cand = response.candidates[0] if response.candidates else None
                if cand and cand.content:
                    contents.append(cand.content)
                contents.append(
                    Content(
                        role="user",
                        parts=[
                            Part(
                                text=(
                                    "Do not call any tools. The host scrolls automatically. "
                                    "Reply with plain text containing only one JSON object."
                                )
                            )
                        ],
                    )
                )
                continue

        raise RuntimeError(
            "Empty model response (expected JSON text, not a tool call)."
        )

    def _collect_rows(
        self,
        analysis: dict,
        w: int,
        h: int,
        pass_index: int,
        result: TableRowsResult,
    ) -> list[dict]:
        rows_on_pass: list[dict] = []
        raw_rows = analysis.get("matching_rows")
        if not isinstance(raw_rows, list):
            return rows_on_pass
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            bbox = self._row_to_pixels(raw, w, h)
            if bbox is None:
                continue
            key = (bbox.get("row_key") or bbox.get("cells_summary") or "").strip().lower()
            if key and key in self.seen_keys:
                continue
            if key:
                self.seen_keys.add(key)
            bbox["pass_index"] = pass_index
            rows_on_pass.append(bbox)
            result.matching_rows.append(bbox)
        return rows_on_pass

    def _row_to_pixels(self, row: dict, w: int, h: int) -> dict | None:
        try:
            y_top = self._denorm(int(row["y_top"]), h)
            y_bottom = self._denorm(int(row["y_bottom"]), h)
            x_left = self._denorm(int(row["x_left"]), w)
            x_right = self._denorm(int(row["x_right"]), w)
        except (KeyError, TypeError, ValueError):
            return None
        if y_bottom <= y_top or x_right <= x_left:
            return None
        return {
            "y_top": y_top,
            "y_bottom": y_bottom,
            "x_left": x_left,
            "x_right": x_right,
            "row_key": str(row.get("row_key") or ""),
            "cells_summary": str(row.get("cells_summary") or ""),
        }

    def denormalize(self, value: int, scale: int) -> int:
        return int(value / 1000 * scale)

    def _denorm(self, value: int, scale: int) -> int:
        return max(0, min(scale - 1, self.denormalize(value, scale)))

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return extract_json_object(text)

    def _build_table_context(self, columns: list[str], analysis: dict) -> dict:
        ctx: dict = {
            "columns_visible": columns,
            "discovery_analysis": analysis,
            "filter_column_number": analysis.get("filter_column_number"),
            "filter_column_name": None,
            "filter_value": analysis.get("filter_value"),
        }
        num = ctx.get("filter_column_number")
        if num is not None and columns:
            try:
                n = int(num)
                if 1 <= n <= len(columns):
                    ctx["filter_column_name"] = columns[n - 1]
            except (TypeError, ValueError):
                pass
        name = self._column_name_from_query(columns)
        if name and columns:
            ctx["filter_column_name"] = name
            ctx["filter_column_number"] = columns.index(name) + 1
        num = ctx.get("filter_column_number")
        cols = ", ".join(columns) if columns else "(unknown)"
        ctx["filter_spec_text"] = (
            f"Filter column #{num} ({ctx.get('filter_column_name')}). "
            f"Layout: {cols}."
        )
        return ctx

    def _column_name_from_query(self, columns: list[str]) -> str | None:
        q = self.query.lower()
        for col in columns:
            if len(col) >= 3 and col.lower() in q:
                return col
        return None

    def _png_hash(self, png_bytes: bytes) -> str:
        return hashlib.sha256(png_bytes).hexdigest()

    def _parse_scroll_request(self, analysis: dict) -> tuple[bool, str | None]:
        """Return (needs_scroll, direction) from model JSON."""
        if analysis.get("should_stop_scrolling") is True:
            return False, None

        needs = analysis.get("needs_scroll")
        if needs is False:
            return False, None
        if needs is not True:
            # Legacy booleans
            if analysis.get("needs_scroll_down") or analysis.get("more_rows_likely_below"):
                return True, "down"
            return False, None

        raw = analysis.get("scroll_direction")
        if raw is None or str(raw).strip().lower() in ("", "null", "none"):
            return False, None
        direction = str(raw).strip().lower()
        if direction not in ("up", "down", "left", "right"):
            return False, None
        return True, direction

    def _should_scroll(self, analysis: dict, consecutive_empty: int) -> bool:
        """Scroll only when the model sets needs_scroll=true with a valid direction."""
        if analysis.get("table_detected") is False:
            print("  Stop: no table detected")
            return False
        if consecutive_empty >= 3:
            print("  Stop: 3 passes with no new rows")
            return False

        needs, direction = self._parse_scroll_request(analysis)
        if not needs or not direction:
            if analysis.get("needs_scroll") is True:
                print("  Stop: needs_scroll=true but scroll_direction missing/invalid")
            else:
                print("  Stop: model set needs_scroll=false (or stop)")
            return False

        self._scroll_direction = direction
        print(f"  Scroll: needs_scroll=true direction={direction}")
        return True

    def _scroll_anchor_xy(self) -> tuple[int, int]:
        anchor = self.table_context.get("scroll_anchor")
        if isinstance(anchor, dict):
            try:
                return int(anchor["x"]), int(anchor["y"])
            except (KeyError, TypeError, ValueError):
                pass
        return (
            int(self.screen_width * 0.35),
            int(self.screen_height * 0.55),
        )

    def _update_scroll_anchor(self, rows: list[dict], screen_height: int) -> None:
        if not rows:
            return
        last = rows[-1]
        x = (int(last["x_left"]) + int(last["x_right"])) // 2
        y = min(screen_height - 1, int(last["y_bottom"]) + 24)
        self.table_context["scroll_anchor"] = {"x": x, "y": y}

    def _scroll(self, direction: str, *, wheel_clicks: int | None = None) -> None:
        
        if direction == "down":
            pyautogui.scroll(-6)
        elif direction == "up":
            pyautogui.scroll(6)
        elif direction == "left":
            pyautogui.hscroll(-1)
        elif direction == "right":
            pyautogui.hscroll(1)
            time.sleep(0.04)

    def _scroll_with_verify(self, before_png: bytes) -> bool:
        """Scroll in the model-requested direction; return False if the screen did not change."""
        before_hash = self._png_hash(before_png)
        self._scroll(self._scroll_direction)
        after = pyautogui.screenshot()
        after_buf = BytesIO()
        after.save(after_buf, format="PNG")
        after_hash = self._png_hash(after_buf.getvalue())
        if after_hash == before_hash:
            print("  Warning: screen unchanged after scroll; retrying with 2× wheel...")
            self._scroll(self._scroll_direction, wheel_clicks=_PAGE_WHEEL_CLICKS * 2)
            time.sleep(_POST_SCROLL_DELAY_S)
            after = pyautogui.screenshot()
            after_buf = BytesIO()
            after.save(after_buf, format="PNG")
            after_hash = self._png_hash(after_buf.getvalue())
        if after_hash == before_hash:
            print("  Error: screen still unchanged after scroll.")
            return False
        if after_hash == self._last_capture_hash:
            print("  Error: scroll returned to a previous view (duplicate pass).")
            return False
        self._last_capture_hash = after_hash
        return True

    def _mark_rows_on_png(self, png_bytes: bytes, rows: list[dict]) -> bytes:
        """Draw semi-transparent row bands on the pass screenshot (physical resolution)."""
        im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = im.size
        for i, row in enumerate(rows):
            try:
                x0, y0 = int(row["x_left"]), int(row["y_top"])
                x1, y1 = int(row["x_right"]), int(row["y_bottom"])
            except (KeyError, TypeError, ValueError):
                continue
            x0, x1 = max(0, min(w - 1, x0)), max(0, min(w - 1, x1))
            y0, y1 = max(0, min(h - 1, y0)), max(0, min(h - 1, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            fill = _ROW_COLORS_RGBA[i % len(_ROW_COLORS_RGBA)]
            outline = (fill[0], fill[1], fill[2], 255)
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=outline, width=3)
        composed = Image.alpha_composite(im, overlay).convert("RGB")
        buf = io.BytesIO()
        composed.save(buf, format="PNG")
        return buf.getvalue()


def find_matching_table_rows(
    query: str,
    *,
    model: str = DEFAULT_TABLE_MODEL,
    max_scroll_passes: int = 8,
    verbose: bool = True,
) -> TableRowsResult:
    """Backward-compatible wrapper for ``vlm.py``."""
    cfg = AgentConfig(
        model, list(TABLE_ROWS_CU_EXCLUDED), [], 1000, 1000, "4"
    )
    return TableRows(cfg, model, query).run(max_scroll_passes=max_scroll_passes)


def save_table_rows_artifacts(
    result: TableRowsResult,
    *,
    out_dir: Path | None = DEFAULT_OUT_DIR,
) -> dict[str, Path]:
    base = out_dir or DEFAULT_OUT_DIR
    base.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    summary_path = base / "table_rows.json"
    summary_path.write_text(
        json.dumps(
            {
                "query": result.query,
                "stop_reason": result.stop_reason,
                "matching_rows": result.matching_rows,
                "passes": [
                    {
                        "pass_index": p.pass_index,
                        "analysis": p.analysis,
                        "rows_on_pass": p.rows_on_pass,
                    }
                    for p in result.passes
                ],
                "column_discovery": result.column_discovery,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    saved["summary_json"] = summary_path

    if result.column_discovery_screenshot:
        cd_path = base / "column_discovery_screenshot.png"
        cd_path.write_bytes(result.column_discovery_screenshot)
        saved["column_discovery_screenshot"] = cd_path

    for p in result.passes:
        idx = p.pass_index
        shot = base / f"pass{idx:02d}_screenshot.png"
        shot.write_bytes(p.screenshot_png)
        saved[f"pass{idx:02d}_screenshot"] = shot
        if p.marked_png:
            marked = base / f"pass{idx:02d}_marked.png"
            marked.write_bytes(p.marked_png)
            saved[f"pass{idx:02d}_marked"] = marked

    result.saved_paths = saved
    return saved


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python table_rows.py '<query>'")
        _sys.exit(1)
    cfg = AgentConfig(
        DEFAULT_TABLE_MODEL,
        list(TABLE_ROWS_CU_EXCLUDED),
        [],
        1000,
        1000,
        "4",
    )
    r = TableRows(cfg, DEFAULT_TABLE_MODEL, _sys.argv[1]).run()
    print(json.dumps({"matching_row_count": len(r.matching_rows), "stop_reason": r.stop_reason}, indent=2))
