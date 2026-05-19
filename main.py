#!/usr/bin/env python3
"""
Interactive launcher for desktop screen agents.

Run from the repo root:
  python screen_agent/main.py

Or:
  cd screen_agent && python main.py

Options:
  1 — find_coordinates   Locate UI points on a screenshot; save marked PNG
  2 — extract_text    Extract visible text (scrolls if needed)
  3 — next_action     Suggest next click/scroll action with coordinates
  4 — table_rows      Find and highlight matching table rows (scrolls if needed)
"""
from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path
from google import genai
from google.genai import types
import time

from agent import AgentConfig
from find_co import FindCoordinates
from extract_text import ExtractText
from next_action import NextAction, NEXT_ACTION_CU_EXCLUDED, build_next_action_function_declarations
from table_rows import TABLE_ROWS_CU_EXCLUDED

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
for _p in (str(REPO_ROOT), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

parser = argparse.ArgumentParser()

from dotenv import load_dotenv

MENU = """
Desktop screen agents
=====================
  1  find_coordinates  — find coordinates on screen and mark them
  2  extract_text   — extract visible text (scroll if needed)
  3  next_action    — next click/scroll step for your task
  4  table_rows     — highlight table rows matching a filter (scroll if needed)
  5  vlm (auto)     — pick the right agent from your query automatically
  q  quit
"""


def _read_query(prompt: str = "Enter your query: ") -> str:
    while True:
        q = input(prompt).strip()
        if q:
            return q
        print("Query cannot be empty. Try again.")


def _read_choice() -> str:
    return input("Select option [1-5, q]: ").strip().lower()




def run_extract_text(query: str) -> None:
    print("\nRunning extract_text (may scroll to find content)...")
    # extract_text = ExtractText(agent_config, model=args.model, query=query)
    # extract_text.run()

def run_next_action(query: str, *, model: str, agent_config: AgentConfig) -> None:
    print("\nCapturing screen and planning next action...")
    result = NextAction(agent_config, model=model, query=query).run()
    print(json.dumps(result.plan, indent=2))
    if "marked_png" not in result.saved_paths:
        print("  (no marked PNG — no click/scroll coordinates in plan)")



def run_table_rows(query: str, *, model: str, agent_config: AgentConfig) -> None:
    from table_rows import DEFAULT_OUT_DIR, TableRows, save_table_rows_artifacts

    print("\nFinding matching table rows on screen (may scroll)...")
    result = TableRows(agent_config, model=model, query=query).run()

    print(
        json.dumps(
            {
                "query": result.query,
                "stop_reason": result.stop_reason,
                "pass_count": len(result.passes),
                "matching_row_count": len(result.matching_rows),
                "matching_rows": result.matching_rows,
            },
            indent=2,
        )
    )

    saved = save_table_rows_artifacts(result, out_dir=DEFAULT_OUT_DIR)
    print("\nSaved files:")
    for key, path in saved.items():
        print(f"  {key}: {path}")

def run_find_coordinates(query: str) -> None:
    pass


_RUNNERS = {
    "1": ("find_coordinates", run_find_coordinates),
    "2": ("extract_text", run_extract_text),
    "3": ("next_action", run_next_action),
    "4": ("table_rows", run_table_rows),
}

def _get_excluded_functions(choice: str) -> list[str]:
    """Computer-use models: exclude predefined tools not needed for this agent."""
    if choice == "3":
        return list(NEXT_ACTION_CU_EXCLUDED)
    if choice == "4":
        return list(TABLE_ROWS_CU_EXCLUDED)
    return ["open_web_browser"]


def _get_function_declarations(choice: str) -> list[types.FunctionDeclaration]:
    """Non-computer-use models: declare tools the host implements in Python."""
    if choice == "1":
        return [
            types.FunctionDeclaration(
                name="click_at",
                description="Click at normalized screen coordinates (0–1000).",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "x": {
                            "type": "INTEGER",
                            "description": "Horizontal coordinate 0–1000 (left to right).",
                        },
                        "y": {
                            "type": "INTEGER",
                            "description": "Vertical coordinate 0–1000 (top to bottom).",
                        },
                    },
                    "required": ["x", "y"],
                },
            )
        ]

    if choice == "3":
        return build_next_action_function_declarations()

    if choice == "2":
        return [
            types.FunctionDeclaration(
                name="scroll_document",
                description=(
                    "Scroll one page at screen center. Use sparingly (max 6 scrolls total). "
                    "Prefer extracting visible text without scrolling when possible."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "direction": {
                            "type": "STRING",
                            "enum": ["up", "down", "left", "right"],
                            "description": "Scroll direction.",
                        },
                    },
                    "required": ["direction"],
                },
            ),
            types.FunctionDeclaration(
                name="scroll_at",
                description=(
                    "Scroll at normalized x,y on the 1000×1000 screenshot. "
                    "Use small magnitude (50–200) for code editors."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "x": {"type": "INTEGER", "description": "0–1000 horizontal."},
                        "y": {"type": "INTEGER", "description": "0–1000 vertical."},
                        "direction": {
                            "type": "STRING",
                            "enum": ["up", "down", "left", "right"],
                        },
                        "magnitude": {
                            "type": "INTEGER",
                            "description": "0–1000; use 50–200 for editors.",
                        },
                    },
                    "required": ["x", "y", "direction", "magnitude"],
                },
            ),
        ]
    if choice == "4":
        return [
            types.FunctionDeclaration(
                name="scroll_document",
                description=(
                    "Scroll one page at screen center to reveal more table rows "
                    "(e.g. when matches may be below the visible area)."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "direction": {
                            "type": "STRING",
                            "enum": ["up", "down", "left", "right"],
                            "description": "Scroll direction.",
                        },
                    },
                    "required": ["direction"],
                },
            ),
        ]

    return []   


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="agent will initialise will take place in this model",
    )

    args = parser.parse_args()
    print(MENU)
    while True:
        choice = _read_choice()
        print(f"choice: {choice}")
        if choice in ("q", "quit", "exit"):
            print("Goodbye.")
            return
        # if choice not in _RUNNERS:
        #     print("Invalid choice. Enter 1, 2, 3, 4, 5, or q.")
        #     continue
        
        name, runner = _RUNNERS[choice]
        print(f"\n>>> {name}")
        query = _read_query()
        
        agent_config = AgentConfig(
            args.model,
            _get_excluded_functions(choice),
            _get_function_declarations(choice),
            1000,
            1000,
            choice,
        )
        try:
            if name == "find_coordinates":
                print(f"waiting for 10 seconds - move to the target screeen")
                time.sleep(10)
                find_coordinates = FindCoordinates(agent_config, model=args.model, query=query)
                find_coordinates.run()
            elif name == "extract_text":
                print("Waiting 10 seconds — focus the editor/window with the text...")
                time.sleep(10)
                agent = ExtractText(
                    agent_config,
                    model=args.model,
                    query=query,
                )
                result = agent.run()
                print(result.as_printable())
            elif name == "next_action":
                print("Waiting 10 seconds — focus the target window...")
                time.sleep(10)
                run_next_action(query, model=args.model, agent_config=agent_config)
            elif name == "table_rows":
                print("Waiting 10 seconds — focus the spreadsheet/table...")
                time.sleep(10)
                run_table_rows(query, model=args.model, agent_config=agent_config)
            else:
                raise ValueError(f"Invalid choice: {choice}")
        except KeyboardInterrupt:
            print("\nCancelled.")
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
        print("\n" + "-" * 40)
        print(MENU)


if __name__ == "__main__":
    main()
