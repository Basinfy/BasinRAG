FROM python:3.11-slim

ARG BASINRAG_PROVIDER_EXTRA=ollama
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 basinrag \
    && mkdir -p /data \
    && chown basinrag:basinrag /data

COPY pyproject.toml README.md LICENSE ./
COPY basinrag ./basinrag
RUN python -m pip install --no-cache-dir ".[api,${BASINRAG_PROVIDER_EXTRA}]"

ENV BASINRAG_STORAGE_DIR=/data/.basinrag
EXPOSE 8000
USER 10001:10001

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/livez', timeout=2)"

CMD ["basinrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
