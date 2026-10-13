# Build context: the CardioSignalAI repository root (see docker-compose.yml),
# so the shared `training/cardiosignal` package can be installed.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY app/backend/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY training/pyproject.toml /opt/cardiosignal/pyproject.toml
COPY training/cardiosignal /opt/cardiosignal/cardiosignal
RUN pip install --no-deps /opt/cardiosignal

COPY app/backend/cardiosignal_api ./cardiosignal_api
COPY app/backend/models ./models
COPY app/backend/data ./data

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

# --proxy-headers: take the client IP from nginx's X-Forwarded-For so rate limiting is per user.
# Safe because the backend is only reachable inside the compose network (not published).
CMD ["uvicorn", "cardiosignal_api.server:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
