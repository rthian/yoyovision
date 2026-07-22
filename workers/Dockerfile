FROM python:3.11-slim

# ffmpeg provides ffprobe/ffmpeg, used by yoyovision_ml.media_validation and
# .preprocessing during the offline analysis pipeline this worker runs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the two local packages this service depends on. `ml` first so its
# layer is reused across api/workers image rebuilds that only touch worker code.
COPY ml/pyproject.toml ml/
COPY ml/src ml/src
COPY workers/pyproject.toml workers/
COPY workers/src workers/src

# Editable installs: docker-compose.yml bind-mounts ./workers and ./ml over
# these copied paths for local dev, so editable mode picks up host-side
# edits without a rebuild. The same image is used unmodified in any
# environment that instead ships code via the built layer (no bind mount).
# Install mediapipe extra so PIPELINE_*_ADAPTER=mediapipe works in containers.
RUN pip install --no-cache-dir -e "./ml[mediapipe]" -e ./workers

# No CMD: docker-compose.yml supplies the queue-specific `celery ... worker`
# command per service (worker-cpu / worker-gpu), per the "Separate CPU and
# GPU task queues" requirement.
