#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is required but was not found in PATH." >&2
    exit 1
fi

if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

if [ ! -x .venv/bin/uvicorn ]; then
    echo "Installing ThinPrompt dependencies..."
    .venv/bin/python -m pip install -e .
fi

if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

set -a
. ./.env
set +a

exec .venv/bin/uvicorn proxy.app:app \
    --host "${PROXY_HOST:-127.0.0.1}" \
    --port "${PROXY_PORT:-8080}"