# Base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Set environment variables for MLflow if needed
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

# Run the app
CMD ["uvicorn", "app.main:router", "--host", "0.0.0.0", "--port", "8000"]
