FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    neo4j \
    psycopg2-binary \
    sqlalchemy \
    fastapi \
    uvicorn \
    python-dotenv \
    pytest

COPY . .

EXPOSE 8003

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
