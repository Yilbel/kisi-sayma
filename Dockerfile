FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 100 opencv-python==4.10.0.84
RUN pip install --no-cache-dir --retries 10 --timeout 100 flask
RUN pip install --no-cache-dir --retries 10 --timeout 100 ultralytics
RUN pip install --no-cache-dir --retries 10 --timeout 100 tf-keras
RUN pip install --no-cache-dir --retries 10 --timeout 100 deepface

COPY . .

CMD ["python", "test.py"]