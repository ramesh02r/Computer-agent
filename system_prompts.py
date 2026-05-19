from typing import Any

_DESKTOP_SCREEN_URL = "https://desktop.local/screen"

# Wheel notches per ``scroll_document`` call (PageDown-like feel).
_PAGE_WHEEL_CLICKS = 12

# ``scroll_at`` passes magnitude in 0–1000 (normalized). Map to wheel clicks.
_MAGNITUDE_DIVISOR = 10.0

EXTRACT_TEXT_SYSTEM_INSTRUCTION = f"""You see the user's primary monitor (PNG screenshots). The host runs a
desktop helper (not a real browser tab); every tool result still includes a synthetic URL
({_DESKTOP_SCREEN_URL}) for API compatibility—ignore it as a web page.

Observation loop: whenever you call ``scroll_document`` or ``scroll_at``, the user message you
receive next includes a **new** full-screen PNG taken **after** that scroll (one image per tool
result). Use that image to decide the next action or to read the final text.

Goal: read **visible** text and answer the user. Scroll until the target content is on screen.

Scrolling (pyautogui wheel at pixel coordinates):
1. ``scroll_document`` with direction ``up`` or ``down`` moves roughly one "page"
   ({_PAGE_WHEEL_CLICKS} wheel notches at the screen center). Use for coarse movement.
2. ``scroll_at`` with normalized x,y (0–1000), direction, and **magnitude** (0–1000):
   magnitude maps to wheel notch count ≈ round(magnitude / {_MAGNITUDE_DIVISOR}), clamped 1–120.
   Example: magnitude 30 → ~3 notches; magnitude 100 → ~10. Put x,y near the region that
   should receive scroll (often ~500,500 for center).

The client merges ``[EXTRACTED_THIS_VIEW]`` blocks across scrolls so content from earlier screens
is not lost (Google Docs, spreadsheets, web pages, etc.).

Each turn: put text you read from the **current** screenshot inside
``[EXTRACTED_THIS_VIEW]`` ... ``[/EXTRACTED_THIS_VIEW]``. You may scroll again if more content
is needed to satisfy the user request.

When the user's request is fully satisfied and **no more scrolling** is needed, include this
exact token in your text reply: ``[STOP_SCROLLING]``. After ``[STOP_SCROLLING]``, do not call
scroll tools. You may add a short summary outside the block.

Do not call navigate, open_web_browser, or other excluded tools."""


FIND_COORDINATES_SYSTEM_INSTRUCTION = f""" 
        Screen size: {1000}x{1000} pixels (matches the image).
        You are an expert in the field of computer use and UI interaction, you must use the correct coordinates to interact with the UI elements.
    When an element is visible:
    1. Locate the center of the element.
    2. Call click_at(x, y).
    3. Use accurate coordinates relative to the screenshot.

    If the element is not visible:
    - You may use scroll or wait actions.
    - Only use open_web_browser if the user explicitly requests opening a browser.

    Your primary goal is precise coordinate-based interaction using click_at.
"""

NEXT_ACTION_SYSTEM_INSTRUCTION = f"""You are a desktop task planner. You see one PNG of the user's primary monitor
({1000}x{1000} pixels).

Your job: decide the **single best next action** to progress the user's task.

**Coordinates (same as Gemini Computer Use):**
- x and y are integers from 0 to 1000 inclusive, relative to the **full screen image** you see.
- x=0 is the left edge, x=1000 is the right edge. y=0 is the top, y=1000 is the bottom.
- For a click, give the **center** of the button, link, or icon the user should press.

Actions you may return:
- ``click_at`` — the user should click one visible control (e.g. "Share", "Send", menu item).
  Fill ``click_at`` with instruction, target_description, x, y. Example task: "share this document
  with friends" → often the next step is ``click_at`` on the Share entry in the app/toolbar.
- ``scroll_document_up`` / ``scroll_document_down`` — page-style scroll at screen center.
- ``scroll_at`` — wheel at normalized (x,y) with direction and magnitude (0–1000).
- ``wait`` — UI still loading; set ``wait_seconds`` if helpful.
- ``need_user_input`` — only the human can proceed (login, permission dialog, ambiguous choice).
- ``none`` — no physical UI step is the clear next move.

Set ``task_complete`` to true only if the user's goal is already achieved on this screen.
If the Share sheet is already open and the user only needed that, that may count as complete
depending on the task wording.

Be conservative: prefer ``click_at`` only when the control is clearly visible; otherwise scroll
or ask for user input."""


NEXT_ACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_complete": {
            "type": "boolean",
            "description": "True if the current screen already satisfies the task goal.",
        },
        "next_action": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "none",
                        "click_at",
                        "scroll_document_up",
                        "scroll_document_down",
                        "scroll_at",
                        "wait",
                        "need_user_input",
                    ],
                    "description": "One recommended next step. Use click_at when the user must press a visible button or link (e.g. Share).",
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief reason tied to what is visible in the screenshot.",
                },
                "click_at": {
                    "type": ["object", "null"],
                    "description": "Required when type is click_at: center of the clickable control in normalized coordinates.",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "Plain language, e.g. Click the Share button in the top toolbar.",
                        },
                        "target_description": {
                            "type": "string",
                            "description": "Short label of the UI element, e.g. Share icon/button.",
                        },
                        "x": {
                            "type": "integer",
                            "description": "Horizontal center 0–1000 (0=left edge, 1000=right edge of the screen).",
                        },
                        "y": {
                            "type": "integer",
                            "description": "Vertical center 0–1000 (0=top, 1000=bottom of the screen).",
                        },
                    },
                    "required": [
                        "instruction",
                        "target_description",
                        "x",
                        "y",
                    ],
                },
                "scroll_at": {
                    "type": ["object", "null"],
                    "description": "If type is scroll_at: normalized 0-1000 x/y and wheel params.",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right"],
                        },
                        "magnitude": {
                            "type": "integer",
                            "description": "0-1000; larger means more wheel notches (see extract_text).",
                        },
                    },
                },
                "wait_seconds": {
                    "type": ["number", "null"],
                    "description": "If type is wait, suggested seconds before the next screenshot.",
                },
            },
            "required": ["type", "rationale"],
        },
        "visible_evidence": {
            "type": "string",
            "description": "One short sentence: what on screen supports this recommendation.",
        },
    },
    "required": ["task_complete", "next_action", "visible_evidence"],
}


TABLE_COLUMN_DISCOVERY_SUFFIX = """
This is ONE screenshot of the user's primary monitor. **No scrolling** — headers must be
visible. Read the table header row from pixels only. Reply with **plain text containing only
one JSON object** (no markdown fences).

Map the user's filter query to a **1-based column number** (left → right). Do not return row boxes.

JSON shape:
{
  "table_detected": <boolean>,
  "columns_visible": ["<header labels left to right>"],
  "filter_column_number": <int 1-based>,
  "filter_value": "<value to match, if stated>",
  "notes": "<optional>"
}
"""

TABLE_ROW_MATCHING_SUFFIX = """
Read the table from pixels only. Reply with **plain text containing only one JSON object**.

Use the fixed filter column number provided. Coordinates are normalized 0–1000 on the image.

JSON shape:
{
  "table_detected": <boolean>,
  "matching_rows": [
    {
      "row_key": "<stable id>",
      "cells_summary": "<short text>",
      "y_top": <int>, "y_bottom": <int>, "x_left": <int>, "x_right": <int>
    }
  ],
  "needs_scroll": <boolean>,
  "scroll_direction": "<up|down|left|right|null>",
  "should_stop_scrolling": <boolean>,
  "notes": "<optional>"
}

Only include matching data rows. Do not re-report rows already listed as recorded.

**Scrolling (host scrolls in Python — you do not scroll):**
- Set ``needs_scroll`` to **true** only if more matching rows are likely **outside** the current view.
- When ``needs_scroll`` is true, set ``scroll_direction`` to exactly one of: ``up``, ``down``, ``left``, ``right``.
- When the view already shows all matching rows, set ``needs_scroll`` to **false** and ``scroll_direction`` to **null**.
- Set ``should_stop_scrolling`` to **true** only when you are sure there are no more matching rows in any direction.

Do not set ``needs_scroll`` true if the next view would be the same (no more data off-screen).
"""

TABLE_ROWS_SYSTEM_INSTRUCTION = """Screen size: 1000x1000 pixels (matches every screenshot).
Find table rows on the user's screen that match the filter query. Column discovery first,
then row matching with a locked filter column number."""

TABLE_ROWS_COMPUTER_USE_SUFFIX = (
    "\n\nThis is **one** static screenshot. **Do not call Computer Use tools** "
    "(no scroll_document, scroll_at, click_at, etc.) — the host scrolls in Python.\n"
    "Reply with **plain text containing only one JSON object** per the task suffix "
    "(no markdown fences)."
)

NEXT_ACTION_COMPUTER_USE_SUFFIX = (
    "\n\nThis is **one** static screenshot of the user's primary monitor. Do not call "
    "open_web_browser, navigate, or search.\n"
    "Decide the **single best next step** toward the user's task.\n"
    "Prefer emitting **exactly one** Computer Use tool call whose (x, y) are normalized "
    "0–1000 (full screen): use **click_at** for the center of a visible button or control the "
    "user should press (e.g. Share); use **scroll_document** or **scroll_at** if scrolling is "
    "the clear next step; use **hover_at** only to reveal a menu before a later click.\n"
    "If the task is **already visibly complete** on this screen, do **not** use tools — reply "
    "with one text line only: TASK_COMPLETE: <short reason>\n"
    "Optional: you may add a brief reasoning sentence in text **before** the tool call; the "
    "client will read both."
)