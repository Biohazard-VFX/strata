FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/strata/training
COPY posttrain/pyproject.toml posttrain/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY posttrain/src ./src
RUN uv sync --frozen --no-dev
COPY posttrain/configs ./configs
ENV TORCH_HOME=/data/torch-home PYTHONPATH="" PYTORCH_ALLOC_CONF=expandable_segments:True
ENTRYPOINT ["uv", "run", "--frozen", "--no-dev", "strata-train"]
CMD ["pipeline", "configs/runpod.json"]
