FROM python:3.12-slim

# Set system environment variables for Python stability inside Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /code

# Install system-level dependencies:
# 1. build-essential & g++ are required to compile FAISS CPU bindings
# 2. git is required by your gitpython wrapper to clone repositories dynamically
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the core application directory
COPY ./app /code/app

# Ensure standard base directories exist inside the container on boot
RUN mkdir -p /code/data/repos /code/data/faiss_indices

# Expose the production port
EXPOSE 8000

# Spin up uvicorn securely with host-binding for cloud routing
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000
