#!/bin/bash

echo "entrypoint.sh"
whoami
which uv
uv run python app.py
