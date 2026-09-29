"""Download Cricsheet Parquet files from TIGZIG."""

from pathlib import Path

import httpx
from loguru import logger
from tqdm import tqdm

from src.config import CRICSHEET_TABLES, RAW_DIR


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Stream-download a file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_bytes(chunk_size):
                f.write(chunk)
                bar.update(len(chunk))

    logger.info(f"Downloaded {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def download_all(force: bool = False) -> dict[str, Path]:
    """Download all Cricsheet tables.  Skip files that already exist unless *force*."""
    paths: dict[str, Path] = {}

    for table_name, url in CRICSHEET_TABLES.items():
        dest = RAW_DIR / f"{table_name}.parquet"

        if dest.exists() and not force:
            logger.info(f"Skipping {table_name} — already exists at {dest}")
            paths[table_name] = dest
            continue

        logger.info(f"Downloading {table_name} from {url}")
        paths[table_name] = download_file(url, dest)

    return paths


if __name__ == "__main__":
    download_all()
