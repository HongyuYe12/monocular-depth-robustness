from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .io import list_images, load_rgb, stable_id, write_csv

MANIFEST_FIELDS = [
    "image_id",
    "source_relpath",
    "variant_id",
    "perturbation",
    "severity",
    "value",
    "image_path",
    "width",
    "height",
]


def deterministic_seed(global_seed: int, image_id: str, value: float) -> int:
    token = f"{global_seed}:{image_id}:{value:g}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def apply_perturbation(
    image: Image.Image,
    kind: str,
    value: float,
    *,
    seed: int = 0,
) -> Image.Image:
    image = image.convert("RGB")
    if kind == "original":
        return image.copy()
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(value)
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=value))
    if kind == "noise":
        rng = np.random.default_rng(seed)
        array = np.asarray(image, dtype=np.float32)
        noisy = np.clip(array + rng.normal(0.0, value, array.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    if kind == "resolution":
        if not 0 < value <= 1:
            raise ValueError("resolution scale must be in (0, 1]")
        width, height = image.size
        small = image.resize(
            (max(1, round(width * value)), max(1, round(height * value))),
            Image.Resampling.BICUBIC,
        )
        return small.resize((width, height), Image.Resampling.BICUBIC)
    raise ValueError(f"unknown perturbation: {kind}")


def _variant_name(kind: str, severity: int, value: float) -> str:
    value_text = f"{value:g}".replace(".", "p")
    return "original" if kind == "original" else f"{kind}-s{severity}-{value_text}"


def build_dataset(input_dir: Path, output_dir: Path, config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    images = list_images(input_dir)
    if not images:
        raise ValueError(f"no supported images found in {input_dir}")

    perturbations = config.get("perturbations", {})
    global_seed = int(config.get("seed", 0))
    generated_dir = output_dir / "images"
    generated_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for source_path in images:
        relative = source_path.relative_to(input_dir).as_posix()
        image_id = stable_id(relative)
        image = load_rgb(source_path)
        variants: list[tuple[str, int, float]] = [("original", 0, 0.0)]
        for kind in ("brightness", "blur", "noise", "resolution"):
            for severity, value in enumerate(perturbations.get(kind, []), start=1):
                variants.append((kind, severity, float(value)))

        for kind, severity, value in variants:
            variant_id = _variant_name(kind, severity, value)
            result = apply_perturbation(
                image,
                kind,
                value,
                seed=deterministic_seed(global_seed, image_id, value),
            )
            destination = generated_dir / image_id / f"{variant_id}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.save(destination, format="PNG", optimize=True)
            rows.append(
                {
                    "image_id": image_id,
                    "source_relpath": relative,
                    "variant_id": variant_id,
                    "perturbation": kind,
                    "severity": severity,
                    "value": f"{value:g}",
                    "image_path": destination.resolve(),
                    "width": result.width,
                    "height": result.height,
                }
            )

    manifest_path = output_dir / "manifest.csv"
    write_csv(manifest_path, rows, MANIFEST_FIELDS)
    snapshot_path = output_dir / "experiment.snapshot.json"
    with snapshot_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic robustness dataset.")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing source images")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset directory")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args.input, args.output, args.config)
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
