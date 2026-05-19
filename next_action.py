"""
Given a screenshot and task description, return the single best next UI action (click/scroll).

Class-based API matches ``FindCoordinates`` and ``ExtractText``:
  agent = NextAction(agent_config, model=..., query=...)
  result = agent.run()
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import pyautogui
from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent import AgentConfig
from utils import (
    DEFAULT_COMPUTER_USE_MODEL,
    collect_text_from_response,
    denormalize_xy,
    draw_markers_on_png,
    extract_json_object,
    generate_content_with_retries,
    iter_response_parts,
)

DEFAULT_NEXT_ACTION_MODEL = DEFAULT_COMPUTER_USE_MODEL
DIR = _PKG_DIR / "next_action_output"
_MODEL_W = 1000
_MODEL_H = 1000

NEXT_ACTION_CU_EXCLUDED: tuple[str, ...] = (
    "open_web_browser",
    "navigate",
    "search",
    "go_back",
    "go_forward",
    "wait_5_seconds",
)


@dataclass
class NextActionResult:
    plan: dict[str, Any]
    screenshot_png: bytes
    full_screenshot_png: bytes
    screen_width: int
    screen_height: int
    saved_paths: dict[str, Path] = field(default_factory=dict)


def build_next_action_function_declarations() -> list[types.FunctionDeclaration]:
    """Tools for non-computer-use models (also wired from main.py choice 3)."""
    return [
        types.FunctionDeclaration(
            name="click_at",
            description="Click at normalized coordinates (0–1000) on the screenshot.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "x": {"type": "INTEGER"},
                    "y": {"type": "INTEGER"},
                },
                "required": ["x", "y"],
            },
        ),
        types.FunctionDeclaration(
            name="scroll_document",
            description="Scroll one page at screen center.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "direction": {
                        "type": "STRING",
                        "enum": ["up", "down", "left", "right"],
                    },
                },
                "required": ["direction"],
            },
        ),
        types.FunctionDeclaration(
            name="scroll_at",
            description="Scroll at x,y with direction and magnitude (0–1000).",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "x": {"type": "INTEGER"},
                    "y": {"type": "INTEGER"},
                    "direction": {
                        "type": "STRING",
                        "enum": ["up", "down", "left", "right"],
                    },
                    "magnitude": {"type": "INTEGER"},
                },
                "required": ["x", "y", "direction", "magnitude"],
            },
        ),
    ]


def make_next_action_agent_config(model: str) -> AgentConfig:
    return AgentConfig(
        model=model,
        excluded_functions=list(NEXT_ACTION_CU_EXCLUDED),
        function_declarations=build_next_action_function_declarations(),
        w0=_MODEL_W,
        h0=_MODEL_H,
        choice="3",
    )


class NextAction:
    def __init__(
        self,
        agent_config: AgentConfig,
        model: str,
        query: str,
        *,
        context: str | None = None,
    ):
        self.agent_config = agent_config
        self.model = model
        self.query = query.strip()
        self.context = (context or "").strip()
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        )
        self.generate_content_config = self.agent_config.get_config()
        self.contents: list[types.Content] = []
        self.screen_width = 0
        self.screen_height = 0
        self.png0: bytes = b""
        self.full_png: bytes = b""

    def run(self) -> NextActionResult:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit(
                "Set GEMINI_API_KEY or GEMINI_API_KEY in the environment (e.g. in a .env file)."
            )

        if DIR.exists():
            shutil.rmtree(DIR)
        DIR.mkdir(parents=True, exist_ok=True)

        self.png0, _, _ = self._fresh_screen_png()
        self.contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=self.png0, mime_type="image/png"),
                    types.Part(text=self._user_message()),
                ],
            )
        ]

        response = generate_content_with_retries(
            self.client,
            model=self.model,
            contents=self.contents,
            config=self.generate_content_config,
        )
        text = (response.text or "").strip() or collect_text_from_response(response).strip()
        plan = self._parse_response(response, text)
        enrich_plan_with_pixel_coordinates(
            plan, screen_width=self.screen_width, screen_height=self.screen_height
        )
        result = NextActionResult(
            plan=plan,
            screenshot_png=self.png0,
            full_screenshot_png=self.full_png,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
        )
        save_next_action_artifacts(result, out_dir=DIR)
        return result

    def _fresh_screen_png(self) -> tuple[bytes, int, int]:
        screenshot = pyautogui.screenshot()
        self.screen_width, self.screen_height = screenshot.width, screenshot.height
        screenshot.save(DIR / "captured_screen.png")
        full_buf = BytesIO()
        screenshot.save(full_buf, format="PNG")
        self.full_png = full_buf.getvalue()
        model_buf = BytesIO()
        resized = screenshot.resize((_MODEL_W, _MODEL_H))
        resized.save(DIR / "resized_1000.png")
        resized.save(model_buf, format="PNG")
        print(
            f"physical: {self.screen_width}x{self.screen_height}, "
            f"model image: {_MODEL_W}x{_MODEL_H}"
        )
        return model_buf.getvalue(), self.screen_width, self.screen_height

    def _user_message(self) -> str:
        lines = [
            f"Screen size: {_MODEL_W}x{_MODEL_H} pixels (matches the image).",
            "",
            "Task to complete:",
            self.query,
        ]
        if self.context:
            lines.extend(["", "Additional context:", self.context])
        return "\n".join(lines)

    def _parse_response(
        self, response: types.GenerateContentResponse, text: str
    ) -> dict[str, Any]:
        payload: dict[str, Any] | None = None
        if _extract_function_calls(response):
            payload = _plan_from_computer_use_function_calls(response)
        if payload is None:
            payload = _plan_from_task_complete_line(text)
        if payload is None and text:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = extract_json_object(text)
            if "parsed_from" not in payload:
                payload["parsed_from"] = "json_text"
        if payload is None:
            raise RuntimeError(
                "Empty model response for next_action (no text and no function_call parts)."
            )
        return payload


def infer_next_action(
    task: str,
    *,
    context: str | None = None,
    model: str = DEFAULT_NEXT_ACTION_MODEL,
    agent_config: AgentConfig | None = None,
) -> NextActionResult:
    """Backward-compatible helper (e.g. ``vlm.py``)."""
    config = agent_config or make_next_action_agent_config(model)
    return NextAction(config, model=model, query=task, context=context).run()


# --- response parsing helpers ---


def _extract_function_calls(
    response: types.GenerateContentResponse,
) -> list[types.FunctionCall]:
    out: list[types.FunctionCall] = []
    for part in iter_response_parts(response):
        if part.function_call and part.function_call.name:
            out.append(part.function_call)
    return out


def _empty_next_action(
    *,
    typ: str,
    rationale: str,
    click_at: dict[str, Any] | None = None,
    scroll_at: dict[str, Any] | None = None,
    wait_seconds: float | None = None,
) -> dict[str, Any]:
    return {
        "type": typ,
        "rationale": rationale,
        "click_at": click_at,
        "scroll_at": scroll_at,
        "wait_seconds": wait_seconds,
    }


def _function_call_to_plan(fc: types.FunctionCall, rationale: str) -> dict[str, Any]:
    name = (fc.name or "").strip()
    args = dict(fc.args or {})
    r = rationale.strip() or f"Model tool: {name}"

    if name in ("click_at", "hover_at"):
        try:
            x = int(round(float(args["x"])))
            y = int(round(float(args["y"])))
        except (KeyError, TypeError, ValueError):
            x, y = 0, 0
        verb = "Click" if name == "click_at" else "Hover"
        return {
            "task_complete": False,
            "visible_evidence": r[:800],
            "next_action": _empty_next_action(
                typ="click_at",
                rationale=r,
                click_at={
                    "instruction": f"{verb} at ({x}, {y}).",
                    "target_description": name,
                    "x": max(0, min(1000, x)),
                    "y": max(0, min(1000, y)),
                },
            ),
        }

    if name == "scroll_document":
        direction = str(args.get("direction", "")).lower()
        if direction == "up":
            typ = "scroll_document_up"
        elif direction == "down":
            typ = "scroll_document_down"
        elif direction in ("left", "right"):
            return {
                "task_complete": False,
                "visible_evidence": r[:800],
                "next_action": _empty_next_action(
                    typ="scroll_at",
                    rationale=r,
                    scroll_at={
                        "x": 500,
                        "y": 500,
                        "direction": direction,
                        "magnitude": 400,
                    },
                ),
            }
        else:
            typ = "none"
        return {
            "task_complete": False,
            "visible_evidence": r[:800],
            "next_action": _empty_next_action(typ=typ, rationale=r),
        }

    if name == "scroll_at":
        try:
            x = int(round(float(args["x"])))
            y = int(round(float(args["y"])))
            direction = str(args["direction"])
            mag = int(round(float(args.get("magnitude", 800))))
        except (KeyError, TypeError, ValueError):
            return {
                "task_complete": False,
                "visible_evidence": r[:800],
                "next_action": _empty_next_action(
                    typ="none", rationale=r + " (invalid scroll_at args)"
                ),
            }
        return {
            "task_complete": False,
            "visible_evidence": r[:800],
            "next_action": _empty_next_action(
                typ="scroll_at",
                rationale=r,
                scroll_at={
                    "x": max(0, min(1000, x)),
                    "y": max(0, min(1000, y)),
                    "direction": direction,
                    "magnitude": max(0, min(1000, mag)),
                },
            ),
        }

    if name == "type_text_at":
        return {
            "task_complete": False,
            "visible_evidence": r[:800],
            "next_action": _empty_next_action(
                typ="need_user_input",
                rationale=r + " (type_text_at: confirm field and text first.)",
            ),
        }

    return {
        "task_complete": False,
        "visible_evidence": f"Unhandled tool {name!r}. {r[:400]}",
        "next_action": _empty_next_action(
            typ="none", rationale=r + f" (raw tool: {name})"
        ),
        "raw_function_call": {"name": name, "args": args},
    }


def _plan_from_computer_use_function_calls(
    response: types.GenerateContentResponse,
) -> dict[str, Any] | None:
    calls = _extract_function_calls(response)
    if not calls:
        return None
    rationale = collect_text_from_response(response).strip()
    meta = [{"name": c.name, "args": dict(c.args or {})} for c in calls]
    plan = _function_call_to_plan(calls[0], rationale)
    plan["parsed_from"] = "computer_use_function_call"
    plan["computer_use_function_calls"] = meta
    return plan


def _plan_from_task_complete_line(text: str) -> dict[str, Any] | None:
    t = text.strip()
    if not t:
        return None
    head, _, rest = t.partition(":")
    if head.strip().upper() != "TASK_COMPLETE":
        return None
    reason = rest.strip() or "Task appears complete on this screen."
    return {
        "task_complete": True,
        "visible_evidence": reason,
        "parsed_from": "task_complete_line",
        "next_action": _empty_next_action(typ="none", rationale=reason),
    }


def _coord_pair_label(kind: str, xn: int, yn: int, px: int, py: int, *, extra: str = "") -> str:
    suffix = f" {extra}" if extra else ""
    return f"{kind} norm({xn},{yn}) px({px},{py}){suffix}"


def enrich_plan_with_pixel_coordinates(
    plan: dict[str, Any], *, screen_width: int, screen_height: int
) -> dict[str, Any]:
    na = plan.get("next_action")
    if not isinstance(na, dict):
        return plan
    w, h = screen_width, screen_height

    ca = na.get("click_at")
    if isinstance(ca, dict):
        try:
            xn, yn = int(ca["x"]), int(ca["y"])
            px, py = denormalize_xy(xn, yn, w, h)
            plan["click_at_pixels"] = {
                "x": px,
                "y": py,
                "screen_width": w,
                "screen_height": h,
                "normalized": {"x": xn, "y": yn},
            }
        except (KeyError, TypeError, ValueError):
            pass

    sa = na.get("scroll_at")
    if isinstance(sa, dict):
        try:
            xn, yn = int(sa["x"]), int(sa["y"])
            px, py = denormalize_xy(xn, yn, w, h)
            plan["scroll_at_pixels"] = {
                "x": px,
                "y": py,
                "screen_width": w,
                "screen_height": h,
                "normalized": {"x": xn, "y": yn},
            }
        except (KeyError, TypeError, ValueError):
            pass

    return plan


def _append_pixel_marker(
    points: list[dict[str, Any]],
    *,
    kind: str,
    xn: int,
    yn: int,
    px: int,
    py: int,
    extra: str = "",
) -> None:
    points.append(
        {
            "x": px,
            "y": py,
            "norm_x": xn,
            "norm_y": yn,
            "label": _coord_pair_label(kind, xn, yn, px, py, extra=extra),
        }
    )


def collect_marker_points(
    plan: dict[str, Any], *, screen_width: int, screen_height: int
) -> list[dict[str, Any]]:
    """Markers on full-resolution capture: position = denormalized px, label shows both pairs."""
    points: list[dict[str, Any]] = []
    na = plan.get("next_action")
    if not isinstance(na, dict):
        return points

    cap = plan.get("click_at_pixels")
    if isinstance(cap, dict) and "x" in cap and "y" in cap:
        norm = cap.get("normalized") or {}
        try:
            xn, yn = int(norm["x"]), int(norm["y"])
            px, py = int(cap["x"]), int(cap["y"])
            ca = na.get("click_at")
            extra = ""
            if isinstance(ca, dict):
                extra = str(ca.get("target_description") or ca.get("instruction") or "")
            _append_pixel_marker(
                points, kind="click", xn=xn, yn=yn, px=px, py=py, extra=extra
            )
        except (KeyError, TypeError, ValueError):
            pass
    else:
        ca = na.get("click_at")
        if isinstance(ca, dict):
            try:
                xn, yn = int(ca["x"]), int(ca["y"])
                px, py = denormalize_xy(xn, yn, screen_width, screen_height)
                extra = str(ca.get("target_description") or ca.get("instruction") or "")
                _append_pixel_marker(
                    points, kind="click", xn=xn, yn=yn, px=px, py=py, extra=extra
                )
            except (KeyError, TypeError, ValueError):
                pass

    sap = plan.get("scroll_at_pixels")
    if isinstance(sap, dict) and "x" in sap and "y" in sap:
        norm = sap.get("normalized") or {}
        try:
            xn, yn = int(norm["x"]), int(norm["y"])
            px, py = int(sap["x"]), int(sap["y"])
            sa = na.get("scroll_at")
            direction = str(sa.get("direction") or "") if isinstance(sa, dict) else ""
            _append_pixel_marker(
                points, kind="scroll", xn=xn, yn=yn, px=px, py=py, extra=direction
            )
        except (KeyError, TypeError, ValueError):
            pass
    else:
        sa = na.get("scroll_at")
        if isinstance(sa, dict):
            try:
                xn, yn = int(sa["x"]), int(sa["y"])
                px, py = denormalize_xy(xn, yn, screen_width, screen_height)
                direction = str(sa.get("direction") or "")
                _append_pixel_marker(
                    points, kind="scroll", xn=xn, yn=yn, px=px, py=py, extra=direction
                )
            except (KeyError, TypeError, ValueError):
                pass

    return points


def collect_norm_marker_points(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Markers on 1000x1000 image: position = normalized coords from the model."""
    points: list[dict[str, Any]] = []
    na = plan.get("next_action")
    if not isinstance(na, dict):
        return points

    for key, kind in (("click_at", "click"), ("scroll_at", "scroll")):
        block = na.get(key)
        if not isinstance(block, dict):
            continue
        try:
            xn, yn = int(block["x"]), int(block["y"])
        except (KeyError, TypeError, ValueError):
            continue
        extra = ""
        if kind == "click":
            extra = str(block.get("target_description") or block.get("instruction") or "")
        elif kind == "scroll":
            extra = str(block.get("direction") or "")
        label = f"{kind} norm({xn},{yn})"
        if extra:
            label = f"{label} {extra}"
        points.append({"x": xn, "y": yn, "norm_x": xn, "norm_y": yn, "label": label})

    return points


def save_next_action_artifacts(
    result: NextActionResult,
    *,
    out_dir: Path | None = None,
    json_path: Path | None = None,
    screenshot_path: Path | None = None,
    marked_png_path: Path | None = None,
    coords_path: Path | None = None,
) -> dict[str, Path]:
    saved: dict[str, Path] = {}
    base = out_dir or DIR
    base.mkdir(parents=True, exist_ok=True)

    plan_path = json_path or base / "next_action.json"
    plan_path.write_text(json.dumps(result.plan, indent=2), encoding="utf-8")
    saved["plan_json"] = plan_path

    captured_path = base / "captured_screen.png"
    if captured_path.is_file():
        saved["captured_screen"] = captured_path
    else:
        full_png = result.full_screenshot_png or result.screenshot_png
        shot_path = screenshot_path or captured_path
        shot_path.write_bytes(full_png)
        saved["captured_screen"] = shot_path

    points = collect_marker_points(
        result.plan,
        screen_width=result.screen_width,
        screen_height=result.screen_height,
    )
    if points:
        if captured_path.is_file():
            captured_bytes = captured_path.read_bytes()
        else:
            captured_bytes = result.full_screenshot_png or result.screenshot_png
        marked_path = marked_png_path or base / "captured_screen_marked.png"
        marked_path.write_bytes(draw_markers_on_png(captured_bytes, points))
        saved["marked_png"] = marked_path

        resized_path = base / "resized_1000.png"
        norm_points = collect_norm_marker_points(result.plan)
        if resized_path.is_file() and norm_points:
            resized_marked = base / "resized_1000_marked.png"
            resized_marked.write_bytes(
                draw_markers_on_png(resized_path.read_bytes(), norm_points)
            )
            saved["resized_1000_marked"] = resized_marked

        na = result.plan.get("next_action")
        coords_payload: dict[str, Any] = {
            "marker_points_pixels": points,
            "marker_points_normalized": norm_points,
            "click_at_pixels": result.plan.get("click_at_pixels"),
            "scroll_at_pixels": result.plan.get("scroll_at_pixels"),
            "next_action_type": na.get("type") if isinstance(na, dict) else None,
            "click_at": na.get("click_at") if isinstance(na, dict) else None,
            "scroll_at": na.get("scroll_at") if isinstance(na, dict) else None,
        }
        coords_file = coords_path or base / "coords.json"
        coords_file.write_text(json.dumps(coords_payload, indent=2), encoding="utf-8")
        saved["coords_json"] = coords_file

    result.saved_paths = saved
    print(f"Saved next_action output to {base}")
    for key, path in saved.items():
        print(f"  {key}: {path}")
    return saved
