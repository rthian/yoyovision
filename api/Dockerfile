FROM python:3.11-slim

# ffmpeg provides ffprobe, which yoyovision_ml.media_validation shells out to
# (read-only, with a timeout) for container/stream metadata during upload
# validation. See api/src/yoyovision_api/services/video_service.py.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the two local packages this service depends on. `ml` first so its
# layer is reused across api/workers image rebuilds that only touch api code.
COPY ml/pyproject.toml ml/
COPY ml/src ml/src
COPY api/pyproject.toml api/
COPY api/src api/src

# Editable installs: docker-compose.yml bind-mounts ./api and ./ml over these
# copied paths for local dev, so editable mode lets `uvicorn --reload` pick
# up host-side edits without a rebuild. The same image is used unmodified in
# any environment that instead ships code via the built layer (no bind mount).
RUN pip install --no-cache-dir -e ./ml -e ./api

EXPOSE 8000

CMD ["sh", "-c", "cd /app/api && alembic upgrade head && uvicorn yoyovision_api.main:app --host 0.0.0.0 --port 8000"]
