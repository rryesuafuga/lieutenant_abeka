#!/usr/bin/env bash
# Re-create everything in output/ from thief_video.mp4 (the original is never modified).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 enhancement/suspect_images.py   # face fusion + stills + summary board (~1 min)
python3 enhancement/enhance_video.py    # full enhanced video + close-up (~10-15 min on 4 cores)
