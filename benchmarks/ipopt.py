"""Access to IPOPT through cyipopt, including the Windows binaries it needs.

On Windows, cyipopt has no wheel on PyPI. It builds against the IPOPT binaries
that COIN-OR publishes, which `fetch` downloads into `benchmarks/.ipopt`. Install
it with `IPOPTWINDIR=<that directory> uv sync --group bench --group ipopt`, and
`import_cyipopt` then makes the IPOPT DLLs visible before importing it. On Linux
and macOS, cyipopt builds against a system IPOPT and needs none of this.
"""

import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from types import ModuleType


VERSION = "3.14.20"
"""The IPOPT release the benchmarks use on Windows."""

ARCHIVE_URL = (
    "https://github.com/coin-or/Ipopt/releases/download/"
    f"releases/{VERSION}/Ipopt-{VERSION}-win64-msvs2022-md.zip"
)
"""The COIN-OR release archive with the Windows binaries."""

ARCHIVE_SHA256 = "1fd1de97d3e8e48c298f6ccf5ac91cbfc417b42c61262c08d193fd14117e9098"
"""The SHA-256 hash of the archive at `ARCHIVE_URL`."""

IPOPT_DIRECTORY = (
    Path(__file__).parent / ".ipopt" / f"Ipopt-{VERSION}-win64-msvs2022-md"
)
"""Where the unpacked binaries end up, which is what `IPOPTWINDIR` points to."""


def fetch() -> None:
    """Download and unpack the IPOPT Windows binaries into `IPOPT_DIRECTORY`."""
    with urllib.request.urlopen(ARCHIVE_URL) as response:
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise RuntimeError(f"The archive has hash {digest}, expected {ARCHIVE_SHA256}.")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        archive.extractall(IPOPT_DIRECTORY.parent)


def import_cyipopt() -> ModuleType:
    """Import cyipopt, first making the IPOPT DLLs visible on Windows.

    Python on Windows does not search `PATH` for the DLLs of an extension module,
    so the binary directory has to be registered with `os.add_dll_directory`.
    """
    if sys.platform == "win32":
        binaries = IPOPT_DIRECTORY / "bin"
        if not binaries.exists():
            raise RuntimeError("Run `python -m benchmarks.ipopt` first.")
        os.add_dll_directory(str(binaries))
    import cyipopt

    return cyipopt


if __name__ == "__main__":
    fetch()
    print(f"Unpacked IPOPT {VERSION} into {IPOPT_DIRECTORY}.")
