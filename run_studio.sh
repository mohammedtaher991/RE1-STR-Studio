#!/bin/sh
cd "$(dirname "$0")"
python3 -m pip install --user -q numpy imageio-ffmpeg 2>/dev/null
python3 -m pip install --user -q pywebview 2>/dev/null || true
exec python3 studio.py "$@"
