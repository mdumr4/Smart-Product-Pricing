import os
import logging
import urllib.request
import concurrent.futures
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("utils")


def download_single_image(url: str, output_folder: Path, timeout: float = 10.0) -> bool:
    """Downloads a single image from URL with timeout."""
    if not isinstance(url, str) or not url.startswith("http"):
        return False

    filename = Path(url).name
    if not filename or len(filename) < 3:
        filename = f"img_{abs(hash(url))}.jpg"

    dest = output_folder / filename
    if dest.exists() and dest.stat().st_size > 0:
        return True

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            with open(dest, "wb") as f:
                f.write(data)
        return True
    except Exception:
        return False


def download_images(
    image_links: List[str], download_folder: str, max_workers: int = 16
) -> int:
    """
    Downloads product images in parallel using ThreadPoolExecutor.
    """
    out_dir = Path(download_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_urls = list(set([u for u in image_links if isinstance(u, str) and u.startswith("http")]))
    logger.info(f"Downloading {len(unique_urls)} images to {out_dir} with {max_workers} threads...")

    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(download_single_image, url, out_dir): url
            for url in unique_urls
        }
        for future in concurrent.futures.as_completed(future_to_url):
            if future.result():
                success_count += 1

    logger.info(f"Downloaded {success_count}/{len(unique_urls)} images successfully.")
    return success_count
