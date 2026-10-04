# CricketMind: the web app (API + chat page) in one container.
#
#   docker build -t cricketmind .                      # small image; data is built on first start
#   docker build -t cricketmind --build-arg BAKE_DATA=1 .   # self-contained image (slow build)
#
# NOT built or run by the author (no Docker on the development machine); see docs/DEPLOYMENT.md.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CRICKETMIND_API_HOST=0.0.0.0 \
    CRICKETMIND_API_PORT=8000 \
    FASTEMBED_CACHE_PATH=/app/data/fastembed_cache

# xgboost needs the OpenMP runtime on Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, against stub sources, so this slow layer is cached until pyproject.toml
# changes. The editable install links to /app/src, which the real code then replaces.
COPY pyproject.toml ./
RUN mkdir src && touch src/__init__.py README.md LICENSE \
    && pip install -e ".[serve]"

COPY src ./src
COPY frontend ./frontend
COPY eval ./eval
COPY docker/entrypoint.sh /entrypoint.sh
COPY README.md LICENSE ./

# Run as an unprivileged user; /app/data is the volume that holds the built data and indexes.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown -R app:app /app /entrypoint.sh \
    && chmod +x /entrypoint.sh
USER app

# Optionally build the data into the image (downloads ~30 MB plus a 130 MB model, ~10-15 min).
ARG BAKE_DATA=0
RUN if [ "$BAKE_DATA" = "1" ]; then python -m src.bootstrap; fi

VOLUME /app/data
EXPOSE 8000

# The first start builds the data (if not baked in), which takes a while: allow for it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --retries=3 \
    CMD python -c "import json,sys,urllib.request as u; d=json.load(u.urlopen('http://127.0.0.1:8000/api/health',timeout=4)); sys.exit(0 if d.get('ready') else 1)"

ENTRYPOINT ["/entrypoint.sh"]
