FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY services/api-gateway/pyproject.toml services/api-gateway/README.md ./
COPY shared ./shared
RUN uv sync --no-dev

COPY services/api-gateway/ .

EXPOSE 8000
CMD ["uv", "run", "python", "main.py"]
