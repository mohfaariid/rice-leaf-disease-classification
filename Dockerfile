# Gunakan image resmi Python 3.10 slim
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Hindari buffering output terminal
ENV PYTHONUNBUFFERED=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install library sistem pendukung image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Salin dependensi terlebih dahulu untuk optimasi caching layer Docker
COPY requirements_web.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Salin kode aplikasi & model terbaik
COPY . /app

# Port default untuk Streamlit (8501) dan FastAPI (8000)
EXPOSE 8501
EXPOSE 8000

# Perintah default untuk menjalankan web app Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
