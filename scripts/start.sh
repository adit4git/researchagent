#!/usr/bin/env sh
set -eu

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

echo "Starting Streamlit on ${HOST}:${PORT}"

exec streamlit run streamlit_app.py \
  --server.address="${HOST}" \
  --server.port="${PORT}" \
  --server.headless=true \
  --browser.gatherUsageStats=false
