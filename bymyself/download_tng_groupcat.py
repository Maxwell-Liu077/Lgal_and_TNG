import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import requests
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

API_KEY = os.environ["TNG_API_KEY"]

BASE_URL = "https://www.tng-project.org/api/TNG50-1/files"

OUTPUT_DIR = Path("/public/share/chenhouzun/TNG50-1/output")

SNAPSHOTS = range(33, 40)

MAX_WORKERS = 4       # Concurrent downloads
MAX_RETRIES = 5       # Retry attempts per file
CHUNK_SIZE = 1024 * 1024  # 1 MB

HEADERS = {
    "api-key": API_KEY,
    "Accept-Encoding": "identity",
}


# ============================================================
# Check downloaded HDF5 file
# ============================================================

def valid_hdf5(path):
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        with h5py.File(path, "r") as f:
            attrs = f["Header"].attrs

            return (
                int(attrs["NumFiles"]) > 0
                and "Ngroups_Total" in attrs
                and "Nsubgroups_Total" in attrs
            )

    except (OSError, KeyError, ValueError):
        return False


# ============================================================
# Download one file
# ============================================================

def download_chunk(snap, chunk_num):

    snap_str = f"{snap:03d}"

    output_folder = OUTPUT_DIR / f"groups_{snap_str}"
    output_folder.mkdir(parents=True, exist_ok=True)

    filename = f"fof_subhalo_tab_{snap_str}.{chunk_num}.hdf5"

    output_file = output_folder / filename
    partial_file = output_folder / (filename + ".part")

    url = (
        f"{BASE_URL}/groupcat-{snap}.{chunk_num}.hdf5"
    )

    # Skip already valid files
    if valid_hdf5(output_file):
        return output_file

    # Remove invalid existing target
    if output_file.exists():
        output_file.unlink()

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            offset = (
                partial_file.stat().st_size
                if partial_file.exists()
                else 0
            )

            headers = HEADERS.copy()

            # Attempt to resume a partial download
            if offset > 0:
                headers["Range"] = f"bytes={offset}-"

            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 180),
            ) as response:

                # Requested range starts beyond server file size
                if response.status_code == 416:
                    partial_file.unlink(missing_ok=True)
                    raise IOError("Invalid resume offset")

                response.raise_for_status()

                # Resume only when server explicitly supports it
                if offset > 0 and response.status_code == 206:

                    content_range = response.headers.get(
                        "Content-Range", ""
                    )

                    match = re.match(
                        r"bytes (\d+)-\d+/(\d+)",
                        content_range,
                    )

                    if not match or int(match.group(1)) != offset:
                        partial_file.unlink(missing_ok=True)
                        raise IOError("Invalid Content-Range")

                    expected_size = int(match.group(2))
                    mode = "ab"

                else:
                    # Server does not support resume:
                    # restart this chunk from the beginning
                    mode = "wb"

                    content_length = response.headers.get(
                        "Content-Length"
                    )

                    expected_size = (
                        int(content_length)
                        if content_length is not None
                        else None
                    )

                with open(partial_file, mode) as f:

                    for data in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):
                        if data:
                            f.write(data)

            # Check file size when provided by server
            if expected_size is not None:
                actual_size = partial_file.stat().st_size

                if actual_size != expected_size:
                    raise IOError(
                        f"Size mismatch: "
                        f"{actual_size} != {expected_size}"
                    )

            # Check HDF5 structure
            if not valid_hdf5(partial_file):
                partial_file.unlink(missing_ok=True)
                raise IOError("Invalid HDF5 file")

            # Mark download complete atomically
            os.replace(partial_file, output_file)

            return output_file

        except Exception as exc:

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed: snapshot {snap}, "
                    f"chunk {chunk_num}"
                ) from exc

            wait_time = min(2 ** attempt, 30)

            print(
                f"\nRetry {attempt}/{MAX_RETRIES}: "
                f"snapshot={snap}, chunk={chunk_num}, "
                f"error={exc}",
                flush=True,
            )

            time.sleep(wait_time)


# ============================================================
# Download a complete group catalog
# ============================================================

def download_snapshot(snap):

    print(f"\n{'=' * 60}")
    print(f"Snapshot {snap}")
    print(f"{'=' * 60}")

    # Download first file to discover NumFiles
    first_file = download_chunk(snap, 0)

    with h5py.File(first_file, "r") as f:
        num_files = int(f["Header"].attrs["NumFiles"])

    print(f"Number of files: {num_files}", flush=True)

    # Download all remaining chunks
    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [
            executor.submit(download_chunk, snap, i)
            for i in range(1, num_files)
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Snapshot {snap}",
            unit="file",
        ):
            future.result()

    # Final check
    output_folder = OUTPUT_DIR / f"groups_{snap:03d}"

    for i in range(num_files):

        filepath = (
            output_folder
            / f"fof_subhalo_tab_{snap:03d}.{i}.hdf5"
        )

        if not valid_hdf5(filepath):
            raise RuntimeError(
                f"Missing or invalid file: {filepath}"
            )

    print(
        f"Snapshot {snap} completed: "
        f"{num_files} files",
        flush=True,
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Simulation: TNG50-1")
    print(f"Snapshots: {list(SNAPSHOTS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Concurrent downloads: {MAX_WORKERS}")

    start_time = time.time()

    for snap in SNAPSHOTS:
        download_snapshot(snap)

    elapsed = (time.time() - start_time) / 3600

    print("\nAll downloads completed!")
    print(f"Elapsed time: {elapsed:.2f} hours")


if __name__ == "__main__":
    main()