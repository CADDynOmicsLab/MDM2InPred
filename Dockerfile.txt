FROM python:3.11-slim

WORKDIR /app

# Install Java and required system tools
RUN apt-get update && \
    apt-get install -y \
    default-jre \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Streamlit port
EXPOSE 8501

# Start Streamlit
CMD ["streamlit", "run", "realsmiles.py", "--server.address=0.0.0.0", "--server.port=8501"]