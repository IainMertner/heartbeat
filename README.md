# heartbeat

For as long as I can remember, I've had an essential tremor. It's a neurodegenerative disease that causes involuntary shaking, most commonly in the hands, and most severely when performing intentional actions. Through this project I have attempted to build a statistical model that learns the pattern of my tremor when drawing a straight, horizontal line, and recreates it.

## Requirements

- Python 3.11 (see `.python-version`)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

With `uv`:

```bash
uv sync
```

This creates a `.venv` and installs everything pinned in `uv.lock`.

Without `uv`, using plain venv + pip:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Running

```bash
uv run main.py
```

(or, with the venv activated, `python main.py`)

This preprocesses/trains as needed and opens a maximized, resizable window showing the scrolling trace. Drag the slider at the bottom of the window to control scroll speed. Close the window or press the OS close button to quit.

If you've added new scans to `raw/cropped` and want to regenerate the processed signals, run the preprocessing step directly first:

```bash
uv run -m scripts.preprocess_data
```

## Building a standalone executable

The project is set up to be packaged with PyInstaller (see `main.spec`). `pyinstaller` is installed in the project's `.venv`, not on your system PATH, so run it via `uv run` (or `.venv\Scripts\pyinstaller` directly):

```powershell
uv run pyinstaller --onefile --windowed `
 --add-data "scripts;scripts" `
 --add-data "raw;raw" `
 --add-data "output;output" `
 main.py
```

This bundles the app plus its raw/processed data into a single `.exe` (output in `dist/`), so it can run without a Python install.

## Project structure

```
main.py                   entry point
scripts/
  preprocess_data.py       raw scans -> processed signals
  train_ar_model.py        fits the AR model used for generation
  generate_col.py          per-column generation logic
  render_col.py            renders a generated column to pixels
  run_app.py               pygame window/event loop
  utils/                   generators, envelope extraction, slider, etc.
raw/                       source scan images (original + cropped)
output/                    processed signals and trained model artifacts
```
