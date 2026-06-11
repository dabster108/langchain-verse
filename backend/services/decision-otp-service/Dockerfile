FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY services/decision-otp-service/pyproject.toml services/decision-otp-service/README.md ./
COPY shared ./shared
RUN uv sync --no-dev

COPY services/decision-otp-service/ .

EXPOSE 8005
CMD ["uv", "run", "python", "main.py"]
