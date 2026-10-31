FROM condaforge/miniforge3:24.11.3-0

ENV PYTHONUNBUFFERED=1 \
    USE_TF=0 \
    USE_FLAX=0 \
    TRANSFORMERS_NO_TF=1 \
    MFA_ROOT_DIR=/home/appuser/.local/share/MFA \
    TRITON_CACHE_DIR=/home/appuser/.cache/triton \
    NATURALNESS_ROOT=/data/naturalness \
    PATH=/opt/mfa/bin:/opt/app-env/bin:${PATH}

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    protobuf-compiler \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN mamba create -y -p /opt/mfa -c conda-forge python=3.11 montreal-forced-aligner \
    && mamba create -y -p /opt/app-env -c conda-forge python=3.11 pip \
    && mamba clean -afy

COPY requirements.txt ./
RUN /opt/app-env/bin/pip install --no-cache-dir -r requirements.txt

COPY app.py research.py speech_core.py batch_research.py deepfake.py textnorm.py stats.py \
     naturalness.py naturalness_cli.py naturalness_ui.py voice_clone.py mms_finetune.py ./
COPY Hindi_Dict_g2g.txt Hindi_All_Acoustic.zip Sanskrit_Dict.txt Sanskrit_Acoustic_Model.zip ./
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /home/appuser/.local/share/MFA /home/appuser/.cache/triton /data/naturalness \
       /opt/clone-env /opt/clone-setup \
    && chown -R appuser:appuser /app /home/appuser/.local /home/appuser/.cache /data \
       /opt/clone-env /opt/clone-setup

USER appuser
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
COPY requirements-clone.txt setup_clone_env.sh /opt/clone-setup/
RUN bash /opt/clone-setup/setup_clone_env.sh /opt/clone-env
ENV VOICE_CLONE_PYTHON=/opt/clone-env/bin/python

RUN /opt/mfa/bin/mfa server init

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]
