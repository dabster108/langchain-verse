FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY services/synthesis-agent/pyproject.toml services/synthesis-agent/README.md ./
COPY shared ./shared
RUN uv sync --no-dev

COPY services/synthesis-agent/ .

EXPOSE 8004
CMD ["uv", "run", "python", "main.py"]
