# Use a lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    curl

# Copy requirements and install Python packages
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Remove build dependencies to slim down final image
RUN apt-get purge -y --auto-remove build-essential gcc git curl && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY . .

# Expose the port Railway expects
EXPOSE 8080

# Start the app using Gunicorn with a single worker to minimize memory usage
CMD ["gunicorn", "-w", "1", "-b", ":8080", "app:app"]
