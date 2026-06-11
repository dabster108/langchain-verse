FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY services/geo-agent/pyproject.toml services/geo-agent/README.md ./
COPY shared ./shared
RUN uv sync --no-dev

COPY services/geo-agent/ .

EXPOSE 8002
CMD ["uv", "run", "python", "main.py"]
