from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FILE_TYPES = {
    "speaker": [("音色文件", "*.speaker.safetensors")],
    "checkpoint": [("模型权重", "*.safetensors *.pt")],
    "codec": [("Codec 权重", "*.pth")],
    "csv": [("CSV 标注", "*.csv")],
    "manifest": [("训练清单", "*.jsonl")],
}


def initial_directory(current: str) -> Path:
    path = Path(current.strip().strip('"')).expanduser()
    while not path.is_dir() and path != path.parent:
        path = path.parent
    return path.resolve()


def choose_path(kind: str, current: str, title: str) -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        options = {"parent": root, "title": title, "initialdir": str(initial_directory(current))}
        if kind == "directory":
            return filedialog.askdirectory(**options, mustexist=True)
        return filedialog.askopenfilename(**options, filetypes=[*FILE_TYPES[kind], ("所有文件", "*.*")])
    finally:
        root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a local file or directory.")
    parser.add_argument("kind", choices=["directory", *FILE_TYPES])
    parser.add_argument("--current", default="")
    parser.add_argument("--title", default="选择路径")
    args = parser.parse_args()
    try:
        print(json.dumps(choose_path(args.kind, args.current, args.title), ensure_ascii=False))
    except Exception as exc:
        print(f"File dialog failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
