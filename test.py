"""Extract and display images returned by the paving-measurement API.

Usage:
    python test.py result.json
    python test.py results.json --no-show
"""

from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image


IMAGE_FIELDS = {
    "satellite_image": "satellite-image.jpg",
    "final_overlay": "final-overlay.png",
    "comparison": "grid-comparison.png",
}


def decode_data_url(data_url: str) -> Image.Image:
    """Decode a ``data:image/...;base64,...`` value into a PIL image."""
    if not data_url.startswith("data:image/") or "," not in data_url:
        raise ValueError("Expected a base64 image data URL.")
    _, encoded = data_url.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def extract_images(result_path: Path, output_dir: Path) -> list[Path]:
    """Extract API response images and return their output paths."""
    response = json.loads(result_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for field, filename in IMAGE_FIELDS.items():
        if field not in response:
            continue
        image = decode_data_url(response[field])
        destination = output_dir / filename
        image.save(destination)
        output_paths.append(destination)
    return output_paths


def show_images(paths: list[Path]) -> None:
    """Display extracted images in a single window when a GUI is available."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(paths), squeeze=False, figsize=(18, 6))
    for axis, path in zip(axes[0], paths):
        axis.imshow(Image.open(path))
        axis.set_title(path.stem.replace("-", " ").title())
        axis.axis("off")
    figure.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_file", nargs="?", type=Path, help="API response JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-show", action="store_true", help="Save images without opening a display window")
    args = parser.parse_args()

    result_path = args.result_file
    if result_path is None:
        result_path = Path("results.json") if Path("results.json").exists() else Path("result.json")
    if not result_path.exists():
        raise SystemExit(f"Result file not found: {result_path}")

    paths = extract_images(result_path, args.output_dir)
    if not paths:
        raise SystemExit("No image fields were found in the API response.")
    print("Saved:")
    for path in paths:
        print(f"- {path}")
    if not args.no_show:
        show_images(paths)


if __name__ == "__main__":
    main()
