FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ARG SERVICE=api-gateway
ARG PORT=8000

WORKDIR /app
ENV PYTHONPATH=/app
ENV SERVICE=${SERVICE}
ENV PORT=${PORT}

COPY pyproject.toml uv.lock ./
COPY shared ./shared
RUN uv sync --frozen --no-dev

COPY services ./services
COPY run_service.py .

EXPOSE ${PORT}
CMD ["uv", "run", "python", "run_service.py"]
