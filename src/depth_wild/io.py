from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Path) -> list[Path]:
    """Return supported images in deterministic relative-path order."""
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )


def load_rgb(path: Path) -> Image.Image:
    """Load an image as RGB and apply its EXIF orientation."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def stable_id(relative_path: str) -> str:
    """Build a readable ID while preventing collisions between equal stems."""
    normalized = relative_path.replace("\\", "/")
    stem = Path(normalized).stem
    readable = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem).strip("_")
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{readable or 'image'}-{digest}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
