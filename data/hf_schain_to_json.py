#!/usr/bin/env python3
"""
Download Hugging Face dataset `leduckhai/S-Chain` and export it to JSONL.

Notes
-----
- The dataset is very large (~34GB).
- JSONL (one JSON object per line) is used so it can be streamed safely.
- Supports exporting all language configs or a single one.
- Optional image saving.

Examples
--------
# Export all configs and splits
python hf_schain_to_json.py --out-dir out

# Export only English train split
python hf_schain_to_json.py --config English --split train --out-dir out

# Stream from HF (less disk usage)
python hf_schain_to_json.py --streaming --out-dir out

# Save images to disk
python hf_schain_to_json.py --config English --split train --save-images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from datasets import (
    load_dataset,
    get_dataset_config_names,
    get_dataset_split_names,
)

DATASET_ID = "leduckhai/S-Chain"


def json_safe(obj: Any) -> Any:
    """Convert objects to JSON-serializable types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    # numpy support
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # PIL image (metadata only unless saved separately)
    try:
        from PIL import Image
        if isinstance(obj, Image.Image):
            return {
                "_type": "PIL.Image",
                "mode": obj.mode,
                "size": list(obj.size),
            }
    except Exception:
        pass

    return str(obj)


def save_image_if_present(
    example: Dict[str, Any],
    key: str,
    images_dir: Path,
    prefix: str,
    idx: int,
) -> Optional[str]:
    """Save image fields to disk if possible and return relative path."""
    if key not in example or example[key] is None:
        return None

    images_dir.mkdir(parents=True, exist_ok=True)
    value = example[key]

    # HF image dict with path
    if isinstance(value, dict) and "path" in value:
        src = Path(value["path"])
        if src.exists():
            dst = images_dir / f"{prefix}_{idx:07d}{src.suffix or '.jpg'}"
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            return str(dst.relative_to(images_dir.parent))

    # PIL image
    try:
        from PIL import Image
        if isinstance(value, Image.Image):
            dst = images_dir / f"{prefix}_{idx:07d}.jpg"
            if not dst.exists():
                value.save(dst, format="JPEG", quality=95)
            return str(dst.relative_to(images_dir.parent))
    except Exception:
        pass

    # raw bytes
    if isinstance(value, (bytes, bytearray)):
        dst = images_dir / f"{prefix}_{idx:07d}.bin"
        if not dst.exists():
            dst.write_bytes(value)
        return str(dst.relative_to(images_dir.parent))

    return None


def iter_examples(ds) -> Iterable[Dict[str, Any]]:
    """Iterate over Dataset or IterableDataset."""
    for ex in ds:
        yield ex


def export_one(
    config: str,
    split: str,
    out_dir: Path,
    streaming: bool,
    save_images: bool,
    limit: Optional[int] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"

    ds = load_dataset(
        DATASET_ID,
        name=config,
        split=split,
        streaming=streaming,
    )

    safe_config = config.replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"{DATASET_ID.replace('/', '__')}__{safe_config}__{split}.jsonl"

    # Try to auto-detect image columns
    image_keys = []
    try:
        if hasattr(ds, "features"):
            for k, v in ds.features.items():
                if v.__class__.__name__.lower() == "image":
                    image_keys.append(k)
    except Exception:
        pass

    # common fallbacks
    for k in ("image", "img"):
        if k not in image_keys:
            image_keys.append(k)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in iter_examples(ds):
            record = dict(ex)

            if save_images:
                for k in image_keys:
                    rel = save_image_if_present(
                        record, k, images_dir,
                        f"{safe_config}_{split}_{k}", n
                    )
                    if rel is not None:
                        record[k] = {"path": rel}

            record = json_safe(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            n += 1
            if limit and n >= limit:
                break

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--config", type=str, default=None,
                        help="Language config (e.g. English). Default: all")
    parser.add_argument("--split", type=str, default=None,
                        help="Split name (train/validation/test). Default: all")
    parser.add_argument("--streaming", action="store_true",
                        help="Stream without full download")
    parser.add_argument("--save-images", action="store_true",
                        help="Save images to disk")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only export first N rows")
    args = parser.parse_args()

    if args.config:
        configs = [args.config]
    else:
        configs = get_dataset_config_names(DATASET_ID)

    for cfg in configs:
        if args.split:
            splits = [args.split]
        else:
            try:
                splits = get_dataset_split_names(DATASET_ID, cfg)
            except Exception:
                splits = ["train"]

        for sp in splits:
            print(f"Exporting {cfg} / {sp} (streaming={args.streaming})")
            path = export_one(
                config=cfg,
                split=sp,
                out_dir=args.out_dir,
                streaming=args.streaming,
                save_images=args.save_images,
                limit=args.limit,
            )
            print(f"  -> {path}")


if __name__ == "__main__":
    main()
