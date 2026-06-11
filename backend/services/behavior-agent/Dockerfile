FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY services/behavior-agent/pyproject.toml services/behavior-agent/README.md ./
COPY shared ./shared
RUN uv sync --no-dev

COPY services/behavior-agent/ .

RUN mkdir -p /app/models

EXPOSE 8003
CMD ["uv", "run", "python", "main.py"]
