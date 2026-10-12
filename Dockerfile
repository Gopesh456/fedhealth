# One image for every service (hospital node, coordinator, dashboard, one-off setup); docker-compose.yml picks the command.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FEDHEALTH_ROOT=/app

WORKDIR /app
# Only code and configuration go into the image - never data, keys or certificates (see .dockerignore).
COPY pyproject.toml ./
COPY fedhealth ./fedhealth
COPY configs ./configs
COPY scripts ./scripts
# editable install keeps the source at /app, so configs/ is found next to the package; gmpy2 = faster PSI
RUN pip install -e ".[dashboard,fast]"

RUN useradd --create-home --uid 10001 app && mkdir -p /data /secrets /certs && chown app /data
USER app

# Real command is supplied per service in docker-compose.yml
CMD ["python", "-c", "print('use docker compose: see README')"]
