FROM python:3.11-slim

# Install uv package manager
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies using uv
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . .

# Expose Chainlit server port
EXPOSE 8000

# Run Chainlit server on port 8000
ENTRYPOINT ["chainlit", "run", "app.py", "--port", "8000", "--host", "0.0.0.0"]
