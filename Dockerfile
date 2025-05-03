# Use a lightweight Python image

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS dependencies first
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install only requirements first (better caching)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Now copy rest of the application
COPY . .

EXPOSE 8080

CMD ["gunicorn", "-b", ":8080", "app:app"]
