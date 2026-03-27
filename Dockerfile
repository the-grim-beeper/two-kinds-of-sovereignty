FROM python:3.11-slim

WORKDIR /app

COPY sovereignty-model/pyproject.toml .
RUN pip install --no-cache-dir .

COPY sovereignty-model/model/ model/
COPY sovereignty-model/app/ app/

EXPOSE ${PORT:-8501}

CMD streamlit run app/dashboard.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
