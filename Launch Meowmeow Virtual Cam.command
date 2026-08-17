#!/bin/bash
cd "$(dirname "$0")"

if [ ! -f .venv/bin/python3 ]; then
    echo "First run — setting up (this takes a minute)..."
    PYTHON=""
    for candidate in python3.13 python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    done
    if [ -z "$PYTHON" ]; then
        echo "Error: this needs Python 3.12 or newer, and none was found on your PATH." >&2
        echo "Install one from https://www.python.org/downloads/ and try again." >&2
        exit 1
    fi
    "$PYTHON" -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -r requirements.txt
fi

.venv/bin/python3 gesture_meme.py --virtual-cam
