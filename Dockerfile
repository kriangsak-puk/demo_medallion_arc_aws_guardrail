FROM python:3.11-slim

# Install uv (standalone binary — fast, no pip needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first (Docker cache layer)
COPY pyproject.toml uv.lock ./

# Install production dependencies only (no dev deps, no project install)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY . .

# Expose Chainlit server port
EXPOSE 8000

# Run Chainlit server on port 8000 via uv
ENTRYPOINT ["uv", "run", "--no-dev", "chainlit", "run", "app.py", "--port", "8000", "--host", "0.0.0.0"]
