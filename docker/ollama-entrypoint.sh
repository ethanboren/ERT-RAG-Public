#!/bin/bash
set -e

ollama serve &

# Attendre que Ollama soit prêt
until curl -s http://localhost:11434/api/tags > /dev/null; do
  echo "⏳ En attente de Ollama..."
  sleep 1
done

# Pull du modèle requis
ollama pull nomic-embed-text

# Garder le conteneur en vie
tail -f /dev/null
