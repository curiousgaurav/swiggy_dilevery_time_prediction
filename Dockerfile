# set the base image
FROM python:3.12-slim

# install lightgbm dependency
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    LIGHTGBM_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1

# set up the working directory
WORKDIR /app

# copy the requirements file
COPY requirements-dockers.txt ./

# install the packages
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements-dockers.txt

# copy the app contents
COPY app.py ./
COPY ./frontend ./frontend
COPY ./models/model.joblib ./models/model.joblib
COPY ./models/preprocessor.joblib ./models/preprocessor.joblib
COPY ./scripts/data_clean_utils.py ./scripts/data_clean_utils.py

# expose the port
EXPOSE 8000

# Run the file using command
CMD [ "python","./app.py" ]