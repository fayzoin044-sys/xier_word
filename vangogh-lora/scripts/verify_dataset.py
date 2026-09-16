"""Verify the prepared training folder with the same ImageFolder loader as training."""
import json
from pathlib import Path


def main():
    from datasets import load_dataset
    root = Path(__file__).resolve().parents[1]
    prepared = root / "data/prepared"
    dataset = load_dataset("imagefolder", data_files={"train": str(prepared / "train/**")},
                           cache_dir=str(root / ".cache/dataset-check"))["train"]
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    groups = {}
    for row in manifest:
        groups.setdefault(row["destination"].split("/")[0], set()).add(row["pixel_sha256"])
    if groups["train"] & groups["val_style"] or groups["eval_photos"] & groups["test_photos"]:
        raise RuntimeError("Data leakage across prepared splits")
    if len(dataset) != len(groups["train"]):
        raise RuntimeError("ImageFolder count does not match manifest")
    for row in dataset:
        if row["image"].mode != "RGB" or not row["text"].strip():
            raise RuntimeError("Invalid image or empty caption")
    result = {"imagefolder_rows": len(dataset), "columns": dataset.column_names,
              "all_images_decode": True, "captions_nonempty": True,
              "train_val_style_disjoint": True, "dev_test_photos_disjoint": True}
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/dataset_loader_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
