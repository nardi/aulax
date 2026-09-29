"""Download the S2MPJ problem files for a pinned commit into a local cache.

Run with `python -m benchmarks.s2mpj.fetch`. The files are not committed, since
the archive is several megabytes and S2MPJ is its own project.
"""

import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path


COMMIT = "fea6a70048eaad28b13a08703ddbfdbf65cd9c30"
"""The S2MPJ commit the benchmarks use, from 24 August 2026."""

FILES = {
    "all_python_problems.tar.gz": (
        "87ce59d70d1ade4c9402bb73c7a4e05ab4033f41ee830cb441e15527e3ce1ae5"
    ),
    "s2mpjlib.py": "4a3a4cb462df754588db332ed499f2a8b5bc8f17db56394e452c4f9b5ad951d3",
}
"""The files to download, with their SHA-256 hashes at `COMMIT`."""

CACHE_DIRECTORY = Path(__file__).parent / ".cache"
"""Where the library and the problem files end up."""

PROBLEM_DIRECTORY = CACHE_DIRECTORY / "python_problems"
"""Where the unpacked problem files end up."""


def download(name: str) -> bytes:
    """Download one file of the pinned commit and check its hash."""
    url = f"https://raw.githubusercontent.com/GrattonToint/S2MPJ/{COMMIT}/{name}"
    with urllib.request.urlopen(url) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != FILES[name]:
        raise RuntimeError(f"{name} has hash {digest}, expected {FILES[name]}.")
    return content


def is_fetched() -> bool:
    """Return whether the cache holds the library and the problem files."""
    return (CACHE_DIRECTORY / "s2mpjlib.py").exists() and PROBLEM_DIRECTORY.exists()


def fetch() -> None:
    """Download and unpack the files into `CACHE_DIRECTORY`."""
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (CACHE_DIRECTORY / "s2mpjlib.py").write_bytes(download("s2mpjlib.py"))
    archive = download("all_python_problems.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        tar.extractall(CACHE_DIRECTORY, filter="data")


if __name__ == "__main__":
    fetch()
    count = len(list(PROBLEM_DIRECTORY.glob("*.py")))
    print(f"Fetched {count} problems into {PROBLEM_DIRECTORY}.")
