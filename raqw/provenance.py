"""Reproducibility manifests for RAQW inputs and outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Iterable


PACKAGE_NAMES = (
    "earthaccess",
    "geopandas",
    "matplotlib",
    "netCDF4",
    "numpy",
    "pandas",
    "pyproj",
    "requests",
    "scipy",
    "shapely",
    "statsmodels",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(directory: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def build_manifest(
    paths: Iterable[Path],
    root: Path | None = None,
    role: str = "input",
) -> list[dict[str, object]]:
    root = Path(root).resolve() if root is not None else None
    records: list[dict[str, object]] = []
    for item in sorted({Path(path).resolve() for path in paths}):
        if not item.is_file():
            raise FileNotFoundError(f"Manifest input is not a file: {item}")
        try:
            display_path = str(item.relative_to(root)) if root else str(item)
        except ValueError:
            display_path = str(item)
        records.append(
            {
                "path": display_path.replace("\\", "/"),
                "role": role,
                "size_bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return records


def write_manifest(
    paths: Iterable[Path],
    output_path: Path,
    root: Path | None = None,
    role: str = "input",
    metadata: dict[str, object] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repository_root = Path(root).resolve() if root is not None else Path.cwd().resolve()
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(repository_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": installed_versions(),
        "metadata": metadata or {},
        "files": build_manifest(paths, root=root, role=role),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
