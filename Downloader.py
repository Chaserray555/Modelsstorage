#!/usr/bin/env python3

import argparse
import re
import shutil
import sys
from pathlib import Path


DEFAULT_DESTINATION = "~/models/"
DEFAULT_INDEX_FILE = "models_index.txt"


def get_huggingface():
    try:
        from huggingface_hub import HfApi, hf_hub_download
        return HfApi, hf_hub_download
    except ImportError:
        print("[!] huggingface_hub is not installed.")
        print()
        print("Install it with:")
        print("  python3 -m pip install --user -U huggingface_hub")
        sys.exit(1)


def parse_model_id(model_id):
    if "@" not in model_id:
        raise ValueError(
            f"Invalid model ID: {model_id}\n"
            f"Expected: publisher/repository@Q4_K_M"
        )

    repo_id, quant = model_id.rsplit("@", 1)

    if "/" not in repo_id:=
        raise ValueError(
            f"Invalid repository: {repo_id}"
        )

    return repo_id.strip(), quant.strip().upper()


def normalize_name(name):
    """
    Make filenames easier to compare.

    Example:
        Q4_K_M
        q4-k-m
        Q4-K-M

    all become:
        q4_k_m
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def find_gguf(files, quant):
    """
    Find the exact requested quantization from the repository.
    """

    gguf_files = [
        f for f in files
        if f.lower().endswith(".gguf")
    ]

    if not gguf_files:
        return None

    wanted = normalize_name(quant)

    matches = []

    for filename in gguf_files:
        normalized = normalize_name(Path(filename).stem)

        if wanted in normalized:
            matches.append(filename)

    if not matches:
        return None

    # Prefer the shortest matching filename.
    # This generally gives us the cleanest model filename.
    matches.sort(key=lambda x: (len(x), x.lower()))

    return matches[0]


def download_model(
    model_id,
    destination,
    HfApi,
    hf_hub_download
):
    repo_id, quant = parse_model_id(model_id)

    print()
    print("=" * 68)
    print(f"[+] Processing: {repo_id}@{quant}")
    print("=" * 68)

    print(f"    -> Looking up repository...")

    api = HfApi()

    try:
        files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="model"
        )

    except Exception as e:
        print(f"    [X] Could not access repository:")
        print(f"        {e}")
        return False

    gguf_file = find_gguf(files, quant)

    if not gguf_file:
        print(f"    [X] No {quant} GGUF file found.")
        print()

        available = [
            f for f in files
            if f.lower().endswith(".gguf")
        ]

        if available:
            print("    Available GGUF files:")
            for filename in available[:20]:
                print(f"        {filename}")

            if len(available) > 20:
                print(
                    f"        ... and "
                    f"{len(available) - 20} more"
                )

        return False

    final_name = Path(gguf_file).name
    final_path = destination / final_name

    print(f"    -> Found exact GGUF:")
    print(f"       {gguf_file}")

    print(f"    -> Destination:")
    print(f"       {final_path}")

    # Already downloaded?
    if final_path.exists():
        if final_path.is_file():
            size_gb = final_path.stat().st_size / (1024 ** 3)

            print()
            print(
                f"    [✓] Already exists "
                f"({size_gb:.2f} GiB)"
            )

            return True

        elif final_path.is_dir():
            print(
                f"    [!] A folder has the same name. "
                f"Removing it..."
            )

            try:
                shutil.rmtree(final_path)
            except Exception as e:
                print(f"    [X] Could not remove folder: {e}")
                return False

    # Temporary directory on the SD card.
    # This prevents the huge GGUF from first filling
    # the Steam Deck's internal storage.
    safe_repo = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        repo_id
    )

    temp_root = destination / ".gguf_download_tmp"
    temp_dir = temp_root / safe_repo

    try:
        temp_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print()
        print("    -> Downloading actual .gguf file...")
        print()

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=gguf_file,
            repo_type="model",
            local_dir=str(temp_dir)
        )

        downloaded_path = Path(downloaded_path)

        if not downloaded_path.exists():
            print(
                "    [X] Download completed but "
                "the file could not be found."
            )
            return False

        if not downloaded_path.is_file():
            print(
                "    [X] Download result is not a file:"
            )
            print(
                f"        {downloaded_path}"
            )
            return False

        # Move the actual GGUF file directly to the
        # root of the models directory.
        print()
        print("    -> Moving GGUF to models folder...")

        shutil.move(
            str(downloaded_path),
            str(final_path)
        )

        # Verify it really is a file and has the
        # correct extension.
        if not final_path.is_file():
            print(
                "    [X] Final result is not a regular file."
            )
            return False

        if final_path.suffix.lower() != ".gguf":
            print(
                "    [X] Final file does not have "
                "a .gguf extension."
            )

            try:
                final_path.unlink()
            except Exception:
                pass

            return False

        size_gb = final_path.stat().st_size / (1024 ** 3)

        print()
        print("    [✓] SUCCESS")
        print(f"        File: {final_path.name}")
        print(f"        Size: {size_gb:.2f} GiB")
        print(f"        Type: GGUF")
        print(f"        Path: {final_path}")

        return True

    except Exception as e:
        print()
        print(f"    [X] Download failed:")
        print(f"        {e}")
        return False

    finally:
        # Delete temporary download material.
        # The actual GGUF was moved out first.
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

        try:
            if temp_root.exists() and not any(temp_root.iterdir()):
                temp_root.rmdir()
        except Exception:
            pass


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Download exact GGUF quantizations "
            "directly from Hugging Face."
        )
    )

    parser.add_argument(
        "--dest",
        "-d",
        default=DEFAULT_DESTINATION,
        help="Destination for GGUF files"
    )

    parser.add_argument(
        "--index",
        "-i",
        default=DEFAULT_INDEX_FILE,
        help="Model index file"
    )

    args = parser.parse_args()

    destination = Path(
        args.dest
    ).expanduser().resolve()

    index_file = Path(
        args.index
    ).expanduser().resolve()

    print("=" * 68)
    print("        GGUF MODEL DOWNLOADER")
    print("        DIRECT .GGUF FILE MODE")
    print("=" * 68)

    print()
    print(f"[*] Destination:")
    print(f"    {destination}")

    print()
    print(f"[*] Model list:")
    print(f"    {index_file}")

    HfApi, hf_hub_download = get_huggingface()

    if not index_file.exists():
        print()
        print(f"[X] Model list does not exist:")
        print(f"    {index_file}")
        sys.exit(1)

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        index_file,
        "r",
        encoding="utf-8"
    ) as f:
        lines = f.readlines()

    successful = 0
    failed = 0

    for line in lines:

        model_id = line.strip()

        # Ignore blank lines
        if not model_id:
            continue

        # Ignore comments
        if model_id.startswith("#"):
            print()
            print(model_id)
            continue

        try:
            result = download_model(
                model_id,
                destination,
                HfApi,
                hf_hub_download
            )

            if result:
                successful += 1
            else:
                failed += 1

        except Exception as e:
            print()
            print(
                f"[X] Unexpected error processing "
                f"{model_id}:"
            )
            print(f"    {e}")
            failed += 1

    print()
    print("=" * 68)
    print("                         COMPLETE")
    print("=" * 68)
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print()
    print("GGUF files are located in:")
    print(destination)
    print("=" * 68)


if __name__ == "__main__":
    main()
