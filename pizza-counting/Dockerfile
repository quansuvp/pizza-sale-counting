FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY wheels/ /wheels/
COPY app/requirements.txt .

RUN pip install --no-cache-dir  --find-links=/wheels -r requirements.txt

COPY app/ .
CMD ["python", "main.py"]

