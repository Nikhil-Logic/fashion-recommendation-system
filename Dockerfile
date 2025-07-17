FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install build tools
RUN apt-get update && \
    apt-get install -y gcc build-essential python3-dev && \
    apt-get clean

# Copy your code
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Default command
CMD ["python", "app.py"]
