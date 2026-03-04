FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy the rest of the application code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the application
CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
