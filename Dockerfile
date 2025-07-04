# Multi-stage build for smaller final image
FROM python:3.11.9-slim-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11.9-slim-bookworm

SHELL ["/bin/bash", "-c"]

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Update PATH to include local packages
ENV PATH=/root/.local/bin:$PATH

# Create working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p data output config

# Copy the modular source code
COPY src/ ./src/
COPY utils/ ./utils/
COPY config/ ./config/
COPY main.py .
COPY download_filter_embed_upsert.py .

# Copy configuration files (if they exist)
COPY .env* ./

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command to run the main application
CMD ["python", "main.py"]