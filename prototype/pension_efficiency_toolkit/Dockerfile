FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY uv.lock* .
COPY pension_toolkit/ ./pension_toolkit/
COPY tests/ ./tests/
COPY scripts/ ./scripts/


# Install dependencies
RUN uv sync --frozen

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "pension_toolkit/ui_streamlit.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
