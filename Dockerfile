FROM condaforge/miniforge3:24.11.3-0

ENV PYTHONUNBUFFERED=1 \
    USE_TF=0 \
    USE_FLAX=0 \
    TRANSFORMERS_NO_TF=1 \
    MFA_ROOT_DIR=/home/appuser/.local/share/MFA \
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

COPY app.py research.py speech_core.py batch_research.py deepfake.py textnorm.py stats.py ./
COPY Hindi_Dict_g2g.txt Hindi_All_Acoustic.zip Sanskrit_Dict.txt Sanskrit_Acoustic_Model.zip ./
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /home/appuser/.local/share/MFA \
    && chown -R appuser:appuser /app /home/appuser/.local

USER appuser

RUN /opt/mfa/bin/mfa server init

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]
