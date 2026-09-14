FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-dockers.txt ./
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements-dockers.txt

COPY app.py ./
COPY frontend ./frontend
COPY scripts/data_clean_utils.py ./scripts/data_clean_utils.py
COPY models/model.joblib ./models/model.joblib

EXPOSE 8000

CMD ["python", "./app.py"]