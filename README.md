# Computer-agent

Desktop screen agents powered by Google Gemini. Capture your primary monitor, send screenshots to a vision model, and run tasks such as finding UI coordinates, extracting text, planning the next click/scroll, or highlighting table rows.

## Prerequisites

- **Python 3.11+** (3.13 tested)
- **Desktop OS with a visible GUI** — **Windows, macOS, or Linux**. The code is not macOS-specific; it uses cross-platform libraries (`mss`, `pyautogui`, `Pillow`).
- A **Google AI / Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey)

**Platform notes (not code restrictions):**

- Screenshots target the **primary monitor** (`mss` monitor index 1, or `pyautogui.screenshot()` depending on the agent). Multi-monitor setups only capture that display unless you change the code.
- Your OS may require permissions for screen capture and simulated input (scroll/click). Examples: **macOS** — Screen Recording and Accessibility in System Settings; **Windows** — allow the terminal if prompted; **Linux** — X11/Wayland display access and sometimes `python3-tk` / display server setup for PyAutoGUI.

The project is **developed and tested mainly on macOS**; scroll speed tuning in `extract_text.py` is labeled for Mac but the same PyAutoGUI calls work on other platforms (you may need different magnitude scaling).

## Setup

1. Clone or open this repository.

2. Create a virtual environment and install dependencies:

```bash
cd Computer-agent
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Add your API key at the **repository root** (parent of `Computer-agent/`), in a file named `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

## Which model to use

| Use case | Recommended model |
|----------|-------------------|
| **General use (best quality)** | **`gemini-3.5`** — preferred for clearer reasoning and better outputs on find-coordinates, extract-text, and vision tasks |
| **Native computer-use tools** | `gemini-2.5-computer-use-preview-10-2025` — use when you need Gemini’s built-in `computer_use` tool (options 3 and 4 with tool calling) |

**Recommendation:** pass **`--model gemini-3.5`** for most queries. If an agent fails or you need strict computer-use tool behavior, switch to the computer-use preview model above.

## How to run

From the `Computer-agent` directory (with the virtual environment activated):

```bash
python main.py --model gemini-3.5-preview
```

Example with the computer-use preview model:

```bash
python main.py --model gemini-2.5-computer-use-preview-10-2025
```


### Interactive menu

After startup you will see:

| Option | Agent | Description |
|--------|--------|-------------|
| **1** | `find_coordinates` | Locate UI elements on screen; saves a marked PNG |
| **2** | `extract_text` | Extract visible text (scrolls if needed) |
| **3** | `next_action` | Suggest the next click or scroll step |
| **4** | `table_rows` | Find and highlight table rows matching your query |
| **q** | — | Quit |

For options **1–4**, you get a short countdown (~10 seconds) to focus the target window before the screenshot is taken.

### Example session

```text
$ python main.py --model gemini-3.5-preview

Desktop screen agents
=====================
  1  find_coordinates  — find coordinates on screen and mark them
  ...

Select option [1-5, q]: 4
Enter your query: highlight rows where status is Pending
```

Outputs (marked images, JSON, etc.) are written under `Computer-agent/` (e.g. `table_rows_output/`, `vlm_output/`).

## Troubleshooting

- **`Set GEMINI_API_KEY...`** — Create `.env` at the repo root or export the variable in your shell.
- **Blank or wrong screen** — Focus the correct window during the countdown; check Screen Recording permission.
- **Model not found** — Confirm the model id in [Gemini models documentation](https://ai.google.dev/gemini-api/docs/models); use `gemini-3.5` or the computer-use preview id exactly as listed in the API.

## Project layout

- `main.py` — Interactive launcher (`--model` required)
- `agent.py` — Shared agent config and Gemini `GenerateContentConfig`
- `find_co.py`, `extract_text.py`, `next_action.py`, `table_rows.py` — Per-task agents
- `utils.py` — Screenshots, API key, JSON parsing, retries
- `requirements.txt` — Python dependencies



Computer-agent/
├── find_co_coordinates/          # results of choice 1 
├── extract_text_output_folder/   # results of choice 2 
├── next_action_output/           # results of choice 3 
├── table_rows_output/            # results of choice 4 
