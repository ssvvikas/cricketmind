"""Serve the API and chat page:  python -m src.api"""

import os

import uvicorn

from src.api.app import create_app
from src.api.settings import ApiSettings

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def main() -> None:
    s = ApiSettings()
    if s.host not in LOOPBACK and not s.access_token:
        print("WARNING: listening on a non-loopback address with no access token. Every visitor "
              "can spend your API credit. Set CRICKETMIND_API_ACCESS_TOKEN.")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("Note: no ANTHROPIC_API_KEY in the environment; chat only works if the SDK finds "
              "another credential (for example an `ant auth login` profile).")
    print(f"CricketMind on http://{s.host}:{s.port}  (loading data in the background)")
    uvicorn.run(create_app(s), host=s.host, port=s.port, log_level="info")


if __name__ == "__main__":
    main()
