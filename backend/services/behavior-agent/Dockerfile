FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch \
    xgboost \
    scikit-learn \
    shap \
    psycopg2-binary \
    sqlalchemy \
    fastapi \
    uvicorn \
    python-dotenv \
    joblib \
    numpy \
    pandas

COPY services/behavior-agent/app/ /app/app/
COPY shared/ /app/shared/
COPY ml/models/ /app/ml/models/

EXPOSE 8004

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
