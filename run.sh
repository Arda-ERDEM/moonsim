#!/bin/bash
# Moon Generator GUI Launcher (for Linux/macOS)

cd "$(dirname "$0")"
source .venv/bin/activate
python -m moon_gen
