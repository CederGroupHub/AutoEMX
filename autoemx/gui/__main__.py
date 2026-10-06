#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch the AutoEMX GUI: ``python -m autoemx.gui [results_folder]``."""

from __future__ import annotations

from typing import Optional

import argparse
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

LAUNCHER_NAME = "AutoEMX GUI"
LAUNCHER_ICON = Path(__file__).with_name("assets") / "autoemx-icon-512.png"


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _set_macos_icon(path: Path) -> bool:
    """Give a file the AutoEMX icon in Finder (best effort; uses the AppKit bridge of osascript)."""
    script = (
        "ObjC.import('AppKit');"
        f"var img = $.NSImage.alloc.initWithContentsOfFile({json.dumps(str(LAUNCHER_ICON))});"
        f"$.NSWorkspace.sharedWorkspace.setIconForFileOptions(img, {json.dumps(str(path))}, 0);"
    )
    try:
        res = subprocess.run(["osascript", "-l", "JavaScript", "-e", script],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return res.returncode == 0 and res.stdout.strip() == "true"


def create_launcher(dest_dir: Optional[str] = None, results_folder: Optional[str] = None) -> Path:
    """
    Write a double-clickable launcher of the GUI (``.command`` on macOS, ``.bat`` on Windows,
    ``.sh`` on Linux) that runs it with the current Python interpreter and AutoEMX installation.
    On macOS the launcher also gets the AutoEMX icon.
    """
    dest = Path(dest_dir).expanduser() if dest_dir else Path.home() / "Desktop"
    if not dest.is_dir():
        dest = Path.home()
    python = sys.executable
    package_parent = str(Path(__file__).resolve().parents[2])
    args = f' "{os.path.abspath(results_folder)}"' if results_folder else ""
    if os.name == "nt":
        path = dest / f"{LAUNCHER_NAME}.bat"
        path.write_text(
            "@echo off\r\n"
            "rem Double-click to open the AutoEMX GUI. Close this window to stop it.\r\n"
            f'set "PYTHONPATH={package_parent};%PYTHONPATH%"\r\n'
            f'"{python}" -m autoemx.gui{args} %*\r\n'
            "pause\r\n",
            encoding="utf-8",
        )
    else:
        path = dest / (f"{LAUNCHER_NAME}.command" if sys.platform == "darwin" else f"{LAUNCHER_NAME}.sh")
        path.write_text(
            "#!/bin/bash\n"
            "# Double-click to open the AutoEMX GUI. Close this window to stop it.\n"
            f'export PYTHONPATH="{package_parent}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
            f'"{python}" -m autoemx.gui{args} "$@"\n',
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        if sys.platform == "darwin":
            _set_macos_icon(path)
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m autoemx.gui",
        description="AutoEMX GUI: quantification, clustering analysis and single-spectrum fits of AutoEMX samples.",
    )
    parser.add_argument("results_folder", nargs="?", default=None,
                        help="Folder containing the sample folders (default: ./Results if present, else none).")
    parser.add_argument("--port", type=int, default=8050, help="Local port (default: 8050).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab.")
    parser.add_argument("--debug", action="store_true", help="Dash debug mode (auto-reload, error pop-ups).")
    parser.add_argument("--create-launcher", nargs="?", const="", default=None, metavar="FOLDER",
                        help="Write a double-clickable launcher (default: on the Desktop) and exit. "
                             "If results_folder is given, the launcher opens it.")
    args = parser.parse_args(argv)

    if args.create_launcher is not None:
        path = create_launcher(args.create_launcher or None, args.results_folder)
        print(f"Launcher written to: {path}\nDouble-click it to open the AutoEMX GUI.")
        return

    import importlib.util

    if importlib.util.find_spec("dash") is None:
        print(
            "The AutoEMX GUI requires Dash.\n"
            "Install it with:  pip install dash\n"
            "Then run:         python -m autoemx.gui",
            file=sys.stderr,
        )
        sys.exit(1)

    url = f"http://127.0.0.1:{args.port}/"
    if _port_in_use(args.port):
        # Most likely the GUI is already running (e.g. launcher double-clicked twice).
        print(f"Port {args.port} is already in use; opening {url}. Use --port to start another instance.")
        if not args.no_browser:
            webbrowser.open(url)
        return

    os.environ.setdefault("MPLBACKEND", "Agg")
    from autoemx.gui.app import JOBS, create_app

    folder = args.results_folder
    if folder is None and os.path.isdir(os.path.join(os.getcwd(), "Results")):
        folder = os.path.join(os.getcwd(), "Results")
    app = create_app(os.path.abspath(folder) if folder else None)

    print(f"AutoEMX GUI running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        # Bound to localhost only: the GUI reads and writes your local files.
        app.run(host="127.0.0.1", port=args.port, debug=args.debug, use_reloader=False)
    finally:
        JOBS.shutdown()


if __name__ == "__main__":
    main()
