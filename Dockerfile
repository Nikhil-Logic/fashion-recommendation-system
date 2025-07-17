FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc build-essential python3-dev && \
    apt-get clean

# Copy your code
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Streamlit runs on (optional)
EXPOSE 8501

# Run the app
CMD ["python", "app.py"]
