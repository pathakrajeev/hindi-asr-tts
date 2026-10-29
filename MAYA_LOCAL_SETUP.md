# Maya 2 Native local setup

Maya runs inside the Streamlit process with the `Maya3TTS` Python API. It does
not require `MAYA_API_BASE_URL`, `MAYA_API_KEY`, or a separate inference
endpoint.

## Requirements

- Linux with Python 3.10–3.12
- NVIDIA GPU and a driver compatible with CUDA 12.8
- At least 12 GB of GPU memory recommended; Maya's documented single-seat
  model-loading peak is approximately 9.2 GB, before CUDA context and other
  models
- Network access on first use to install packages and download model weights

Native Windows is not a supported Maya runtime. On Windows, use WSL2 with
working NVIDIA GPU passthrough or deploy the app on Linux GPU hardware.

## Install and run

The project `requirements.txt` pins the Maya-compatible Torch and Transformers
versions and includes the CUDA 12.8 PyTorch index and Maya package.
Indic Parler-TTS is not included because it requires Transformers 4.46.1,
which cannot coexist with Maya's Transformers 5.6.2 requirement.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

In Hugging Face Spaces, select GPU hardware for the Space and rebuild the
Docker image after changing dependencies. The Dockerfile installs the same
requirements. CPU Spaces can run other models, but Maya will report that CUDA
is required. Maya's Triton kernels cache under `/home/appuser/.cache/triton`,
which the Dockerfile creates and assigns to the app user; this avoids trying
to write cache files into the read-only Python `site-packages` directory.

In Research mode, select **Maya 2 Native (local GPU)** under TTS models and
choose a voice. The first run downloads and caches model assets. Maya is loaded
only when selected; the **Free models after use** option closes its runtime
after synthesis to release GPU memory.
