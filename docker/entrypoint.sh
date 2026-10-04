#!/bin/sh
# Build whatever is missing (instant when the data is already there), then serve.
set -e
python -m src.bootstrap
exec python -m src.api
