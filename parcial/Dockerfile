FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv libgl1 libglib2.0-0 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /workspace
COPY requirements.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir pip==25.1.1 setuptools==80.9.0 \
    && python -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
       --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY configs ./configs
RUN python -m pip install --no-deps --no-build-isolation -e .
ENTRYPOINT ["python", "-m", "neural_exam"]
CMD ["--help"]

FROM python:3.12-slim-bookworm AS reports
RUN apt-get update && apt-get install -y --no-install-recommends \
    latexmk texlive-latex-extra texlive-fonts-recommended texlive-lang-spanish poppler-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/reports
ENTRYPOINT ["python", "/workspace/scripts/compile_reports.py"]
