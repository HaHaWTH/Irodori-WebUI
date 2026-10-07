"""Path fields with desktop dialogs and buttons."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import gradio as gr

from tts_engine import clean_path


def select_path(current: str, kind: str, title: str):
    import json

    command = [sys.executable, "-B", "-X", "utf8", str(Path(__file__).with_name("select_path.py")),
               kind, "--current", current or "", "--title", title]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"Dialog process exited with code {result.returncode}.")
        selected = json.loads(result.stdout)
        return selected if selected else gr.skip()
    except (OSError, ValueError, RuntimeError) as exc:
        raise gr.Error(f"Path selection failed: {exc}") from exc


def open_directory(value: str):
    if not value or not clean_path(value):
        raise gr.Error("Select a directory first.")
    path = Path(clean_path(value)).expanduser().resolve()
    if not path.is_dir():
        raise gr.Error(f"Directory does not exist: {path}")
    try:
        if sys.platform == "win32":
            os.startfile(str(path), "explore")
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
    except OSError as exc:
        raise gr.Error(f"Could not open directory: {exc}") from exc


def path_field(label: str, *, value: str = "", kind: str = "directory", info: str | None = None,
               interactive: bool = True, mode: gr.Radio | None = None):
    with gr.Row(elem_classes="path-row"):
        field = gr.Textbox(label=label, value=value, info=info, interactive=interactive,
                           lines=1, max_lines=2, scale=5, min_width=180)
        if interactive:
            browse = gr.Button("选择文件夹" if kind == "directory" else "选择文件", scale=0,
                               min_width=104, elem_classes="path-button")
            if mode is None:
                def choose(current):
                    return select_path(current, kind, label)
                browse.click(choose, field, field, concurrency_id="file-dialog", concurrency_limit=1,
                             api_name=False, show_progress="hidden")
            else:
                def choose_continuation(current, training_mode):
                    folder = training_mode == "lora"
                    return select_path(current, "directory" if folder else kind,
                                       "继续训练检查点" if folder else label)
                browse.click(choose_continuation, [field, mode], field, concurrency_id="file-dialog",
                             concurrency_limit=1, api_name=False, show_progress="hidden")
                mode.change(lambda selected: gr.update(value="选择文件夹" if selected == "lora" else "选择文件"),
                            mode, browse, queue=False, api_name=False)
        if kind == "directory":
            open_button = gr.Button("打开文件夹", scale=0, min_width=104, elem_classes="path-button")
            open_button.click(open_directory, field, queue=False, api_name=False)
    return field
