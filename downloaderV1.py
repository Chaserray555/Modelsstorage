#!/usr/bin/env python3
"""
GGUF Model Downloader
Optimized for Concurrency, Reliability, and Speed.
"""

import argparse
import concurrent.futures
import logging
import re
import shutil
import sys
import time
from functools import wraps
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
DEFAULT_DESTINATION = "~/models/"
DEFAULT_INDEX_FILE = "models_index.txt"
MAX_CONCURRENT_DOWNLOADS = 3
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

# Set up thread-safe logging to replace standard print statements
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("GGUF_Downloader")


def with_retries(max_retries: int = MAX_RETRIES, backoff: int = RETRY_BACKOFF_SECONDS):
    """Decorator to retry network-dependent functions upon failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_retries:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempts} failed: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
        return wrapper
    return decorator


def get_huggingface() -> Tuple[Any, Any]:
    """Dynamically import and return Hugging Face hub dependencies."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        return HfApi, hf_hub_download
    except ImportError:
        logger.critical("huggingface_hub is not installed.")
        logger.info("Install it with: python3 -m pip install --user -U huggingface_hub")
        sys.exit(1)


def parse_model_id(model_id: str) -> Tuple[str, str]:
    """Parse the model identifier into repository and quantization type."""
    if "@" not in model_id:
        raise ValueError(f"Invalid model ID: {model_id} (Expected: publisher/repository@Q4_K_M)")

    repo_id, quant = model_id.rsplit("@", 1)

    if "/" not in repo_id:
        raise ValueError(f"Invalid repository format: {repo_id}")

    return repo_id.strip(), quant.strip().upper()


def normalize_name(name: str) -> str:
    """Normalize filenames for easier substring matching."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def check_disk_space(path: Path, required_bytes: int) -> bool:
    """Ensure the target drive has enough space for the download."""
    try:
        # Check space on the drive where the destination resides
        total, used, free = shutil.disk_usage(path.resolve().parent)
        # Require an extra 1GB buffer
        return free > (required_bytes + (1024 ** 3))
    except Exception as e:
        logger.warning(f"Could not verify disk space: {e}")
        return True  # Proceed anyway if we can't check


@with_retries()
def get_gguf_metadata(api: Any, repo_id: str, quant: str) -> Optional[Tuple[str, int]]:
    """
    Find the exact requested quantization and its file size.
    Requires fetching full repository metadata.
    """
    try:
        model_info = api.model_info(repo_id=repo_id, files_metadata=True)
    except Exception as e:
        logger.error(f"Could not access repository {repo_id}: {e}")
        return None

    gguf_files = [f for f in model_info.siblings if f.rfilename.lower().endswith(".gguf")]
    
    if not gguf_files:
        return None

    wanted = normalize_name(quant)
    matches = []

    for f_info in gguf_files:
        filename = f_info.rfilename
        normalized = normalize_name(Path(filename).stem)
        if wanted in normalized:
            matches.append((filename, f_info.size))

    if not matches:
        return None

    # Prefer the shortest matching filename
    matches.sort(key=lambda x: (len(x[0]), x[0].lower()))
    return matches[0]


def process_model(model_id: str, destination: Path, api_class: Any, download_func: Any) -> bool:
    """Core download logic executed by the thread pool."""
    try:
        repo_id, quant = parse_model_id(model_id)
    except ValueError as e:
        logger.error(str(e))
        return False

    api = api_class()
    logger.info(f"[{repo_id}@{quant}] Looking up repository metadata...")

    metadata = get_gguf_metadata(api, repo_id, quant)
    if not metadata:
        logger.error(f"[{repo_id}@{quant}] No matching GGUF file found.")
        return False

    gguf_file, file_size_bytes = metadata
    size_gb = file_size_bytes / (1024 ** 3) if file_size_bytes else 0
    final_path = destination / Path(gguf_file).name

    logger.info(f"[{repo_id}@{quant}] Found exact match: {gguf_file} ({size_gb:.2f} GiB)")

    # Pre-flight Check 1: Existence
    if final_path.exists():
        if final_path.is_file():
            existing_size_gb = final_path.stat().st_size / (1024 ** 3)
            logger.info(f"[{repo_id}@{quant}] ALREADY EXISTS ({existing_size_gb:.2f} GiB). Skipping.")
            return True
        elif final_path.is_dir():
            logger.warning(f"[{repo_id}@{quant}] Directory blocking file path. Removing...")
            shutil.rmtree(final_path, ignore_errors=True)

    # Pre-flight Check 2: Disk Space
    if file_size_bytes and not check_disk_space(destination, file_size_bytes):
        logger.error(f"[{repo_id}@{quant}] Insufficient disk space. Requires {size_gb:.2f} GiB.")
        return False

    # Isolate temporary files to prevent cross-thread collisions
    safe_repo = re.sub(r"[^A-Za-z0-9._-]+", "_", repo_id)
    temp_root = destination / ".gguf_download_tmp"
    temp_dir = temp_root / f"{safe_repo}_{quant}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"[{repo_id}@{quant}] Commencing download...")
        
        # Download straight to the isolated temp directory on the SD card
        downloaded_path_str = download_func(
            repo_id=repo_id,
            filename=gguf_file,
            repo_type="model",
            local_dir=str(temp_dir)
        )
        downloaded_path = Path(downloaded_path_str)

        if not downloaded_path.exists() or not downloaded_path.is_file():
            logger.error(f"[{repo_id}@{quant}] File missing or invalid after download.")
            return False

        logger.info(f"[{repo_id}@{quant}] Download complete. Moving to root destination...")
        shutil.move(str(downloaded_path), str(final_path))

        # Final Integrity Verification
        if not final_path.is_file() or final_path.suffix.lower() != ".gguf":
            logger.error(f"[{repo_id}@{quant}] Invalid final file state.")
            final_path.unlink(missing_ok=True)
            return False

        logger.info(f"[{repo_id}@{quant}] SUCCESS -> {final_path.name}")
        return True

    except Exception as e:
        logger.error(f"[{repo_id}@{quant}] Download failed: {e}")
        return False

    finally:
        # Cleanup isolated temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        try:
            if temp_root.exists() and not any(temp_root.iterdir()):
                temp_root.rmdir()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="High-Performance GGUF Downloader for Hugging Face."
    )
    parser.add_argument(
        "--dest", "-d",
        default=DEFAULT_DESTINATION,
        help="Destination directory for GGUF files"
    )
    parser.add_argument(
        "--index", "-i",
        default=DEFAULT_INDEX_FILE,
        help="Text file containing model IDs to download"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=MAX_CONCURRENT_DOWNLOADS,
        help="Number of concurrent downloads"
    )
    args = parser.parse_args()

    destination = Path(args.dest).expanduser().resolve()
    index_file = Path(args.index).expanduser().resolve()

    logger.info("=" * 50)
    logger.info(" GGUF MODEL DOWNLOADER - CONCURRENT EDITION")
    logger.info("=" * 50)
    logger.info(f"Destination: {destination}")
    logger.info(f"Index File : {index_file}")
    logger.info(f"Workers    : {args.workers}")

    HfApi, hf_hub_download = get_huggingface()

    if not index_file.is_file():
        logger.critical(f"Index file missing: {index_file}")
        sys.exit(1)

    destination.mkdir(parents=True, exist_ok=True)

    with open(index_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse and clean the task list
    tasks = []
    for line in lines:
        cleaned = line.strip()
        if cleaned and not cleaned.startswith("#"):
            tasks.append(cleaned)

    if not tasks:
        logger.warning("No valid models found in the index file.")
        sys.exit(0)

    logger.info(f"Found {len(tasks)} models to process. Starting pool...")
    
    successful = 0
    failed = 0

    # Execute downloads concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_model = {
            executor.submit(process_model, model_id, destination, HfApi, hf_hub_download): model_id
            for model_id in tasks
        }

        for future in concurrent.futures.as_completed(future_to_model):
            model_id = future_to_model[future]
            try:
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as exc:
                logger.error(f"[{model_id}] generated an unhandled exception: {exc}")
                failed += 1

    logger.info("=" * 50)
    logger.info(" COMPLETE")
    logger.info("=" * 50)
    logger.info(f" Successful: {successful}")
    logger.info(f" Failed    : {failed}")
    logger.info(f" Location  : {destination}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

