FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# App code
COPY . .

# Default: run the daemon
ENTRYPOINT ["python", "-m", "orchestrator.cli"]
CMD ["daemon", "--interval", "6"]
