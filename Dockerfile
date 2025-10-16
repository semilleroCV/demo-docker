# Use a slim Python base image
FROM python:3.12-slim

# Install uv from its official GitHub Container Registry
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory inside the container
WORKDIR /app

# Copy your application files into the container
COPY . /app

# Install application dependencies using uv sync
# --frozen ensures that the dependencies are installed based on a uv.lock file,
# providing deterministic builds.
# --no-cache prevents uv from using its cache, which can be useful in CI/CD environments.
RUN uv sync --frozen --no-cache


# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command (can be overridden)
CMD ["uv", "run", "python", "train.py"]