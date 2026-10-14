# HealthGraph inference service: FastAPI (/predict, /docs) + Gradio UI (/ui) on port 7860.
# Port 7860 and UID 1000 follow the Hugging Face Spaces Docker SDK conventions.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    HEALTHGRAPH_BUNDLE=/app/artifacts/healthgat_uci.pt

RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-serve.txt

COPY healthgraph/ healthgraph/
COPY app/ app/
COPY artifacts/healthgat_uci.pt artifacts/healthgat_uci.pt
RUN chown -R user:user /app
USER user

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
