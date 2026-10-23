FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

ENV USE_TF=0
ENV USE_FLAX=0
ENV PYTHONUNBUFFERED=1
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
