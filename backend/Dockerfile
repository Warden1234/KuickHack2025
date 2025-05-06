# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files to the container
COPY . .

# Expose the port
EXPOSE 8080

# Command to run the app using Gunicorn
CMD ["hypercorn", "app:app", "--bind", "0.0.0.0:8080"]
