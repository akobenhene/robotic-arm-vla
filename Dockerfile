# Linux demo image (MuJoCo + LeRobot). Prefer CPU for smoke; GPU optional later.
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=osmesa \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev \
    libglib2.0-0 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt streamlit pyyaml

COPY . .

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app_streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]
