FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY services/velocity-agent/pyproject.toml services/velocity-agent/README.md ./
COPY shared ./shared
RUN uv sync --no-dev

COPY services/velocity-agent/ .

EXPOSE 8001
CMD ["uv", "run", "python", "main.py"]
