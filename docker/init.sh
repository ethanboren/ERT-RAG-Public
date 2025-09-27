#!/bin/bash
set -e

# Start cron service
service cron start

# Wait for Ollama service
until curl -s http://ollama:11434/api/tags > /dev/null; do
  echo "Waiting for Ollama..."
  sleep 2
done

# Check if database exists and is populated
echo "Initializing vector database..."
python main.py || { echo "Failed to initialize database"; exit 1; }

# Start Streamlit
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0