#!/usr/bin/env bash
# Build the separate IndicF5 voice-cloning environment and print the VOICE_CLONE_PYTHON to use.
#   bash setup_clone_env.sh [target_dir]      (default /opt/clone-env; needs ffmpeg and network)
set -euo pipefail
TARGET="${1:-/opt/clone-env}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v uv >/dev/null 2>&1 || pip install --no-cache-dir uv
uv venv --python 3.10 "$TARGET"
uv pip install --python "$TARGET/bin/python" --no-cache -r "$HERE/requirements-clone.txt"
"$TARGET/bin/python" -c "import f5_tts, pydub, soundfile, transformers, torch; print('clone env OK: transformers', transformers.__version__, '| torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
echo "export VOICE_CLONE_PYTHON=$TARGET/bin/python"
