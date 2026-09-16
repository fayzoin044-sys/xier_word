"""Audit the original zip and create disjoint, deterministic LoRA splits."""
import argparse
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image, ImageOps

STYLE = "a painting in the style of Vincent van Gogh"
SPLITS = ("trainA", "testA", "trainB", "testB")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare(archive, output, seed=42, holdout=40, eval_count=24):
    if output.exists():
        raise ValueError(f"Output already exists; use a new directory: {output}")
    if holdout < 1 or eval_count < 1:
        raise ValueError("holdout and eval_count must be positive")
    records, errors = [], []
    with ZipFile(archive) as z:
        for info in sorted(z.infolist(), key=lambda x: x.filename):
            path = PurePosixPath(info.filename)
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            split = path.parent.name
            if split not in SPLITS:
                continue
            try:
                raw = z.read(info)
                with Image.open(io.BytesIO(raw)) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image.load()
                    size = image.size
                    digest = hashlib.sha256(str(size).encode() + image.tobytes()).hexdigest()
                records.append(dict(source=info.filename, split=split, width=size[0], height=size[1],
                                    sha256=hashlib.sha256(raw).hexdigest(), pixel_sha256=digest))
            except Exception as exc:
                errors.append({"source": info.filename, "error": str(exc)})
        if errors:
            raise ValueError(f"Unreadable images (no dataset written): {errors[:10]}")
        groups = {s: [r for r in records if r["split"] == s] for s in SPLITS}
        if not all(groups.values()):
            raise ValueError("Expected nonempty trainA, testA, trainB, testB")
        hashes = {s: {r["pixel_sha256"] for r in groups[s]} for s in SPLITS}
        unique_a = {r["pixel_sha256"]: r for r in groups["trainA"]}
        painting_ids = sorted(unique_a)
        random.Random(seed).shuffle(painting_ids)
        if len(painting_ids) <= holdout:
            raise ValueError("Not enough unique paintings for requested holdout")
        val_ids = set(painting_ids[:holdout])
        # Test photos are held out from selection of development photos, even if duplicates exist.
        unique_b = {r["pixel_sha256"]: r for r in groups["trainB"]
                    if r["pixel_sha256"] not in hashes["testB"]}
        photo_ids = sorted(unique_b)
        random.Random(seed).shuffle(photo_ids)
        if len(photo_ids) < eval_count:
            raise ValueError("Not enough disjoint development photos")
        chosen = {
            "train": [unique_a[h] for h in painting_ids if h not in val_ids],
            "val_style": [unique_a[h] for h in painting_ids if h in val_ids],
            "eval_photos": [unique_b[h] for h in photo_ids[:eval_count]],
            "test_photos": list({r["pixel_sha256"]: r for r in groups["testB"]}.values()),
            "test_style_independent": list({r["pixel_sha256"]: r for r in groups["testA"]
                                           if r["pixel_sha256"] not in hashes["trainA"]}.values()),
        }
        output.mkdir(parents=True)
        manifest = []
        for target, rows in chosen.items():
            folder = output / target
            folder.mkdir()
            metadata = []
            for row in rows:
                filename = row["pixel_sha256"] + ".png"
                with Image.open(io.BytesIO(z.read(row["source"]))) as image:
                    ImageOps.exif_transpose(image).convert("RGB").save(folder / filename)
                manifest.append({**row, "destination": f"{target}/{filename}"})
                if target == "train":
                    metadata.append({"file_name": filename, "text": STYLE})
            if target == "train":
                (folder / "metadata.jsonl").write_text(
                    "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metadata), encoding="utf-8")
        with open(archive, "rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        report = {
            "archive_sha256": digest.hexdigest(), "seed": seed,
            "source_counts": {s: len(groups[s]) for s in SPLITS},
            "unique_pixel_counts": {s: len(hashes[s]) for s in SPLITS},
            "sizes": dict(Counter(f'{r["width"]}x{r["height"]}' for r in records)),
            "train_test_pixel_overlap": {"A": len(hashes["trainA"] & hashes["testA"]),
                                          "B": len(hashes["trainB"] & hashes["testB"])},
            "prepared_counts": {s: len(v) for s, v in chosen.items()},
            "caption_mode": "style-only bootstrap; add reviewed content captions before quality experiments",
            "notes": ["Pixel hashes catch exact decoded duplicates, not all near-duplicates.",
                      "Original testA overlaps trainA; independent remainder is too small for reliable style metrics.",
                      "val_style is a held-out style reference, never a paired target for photos."]
        }
        write_json(output / "audit.json", report)
        write_json(output / "manifest.json", manifest)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", type=int, default=40)
    parser.add_argument("--eval-count", type=int, default=24)
    args = parser.parse_args()
    prepare(args.zip, args.output, args.seed, args.holdout, args.eval_count)


if __name__ == "__main__":
    main()
