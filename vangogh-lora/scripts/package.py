"""Package code and audit reports, excluding datasets, environments and weights."""
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT.parent / "vangogh-lora-v1.zip"
    roots = [ROOT / p for p in ("scripts", "configs", "vendor", "tests", "reports")]
    files = [ROOT / p for p in ("README.md", "EVALUATION.md", "requirements.txt", ".gitignore")]
    files += [f for folder in roots for f in folder.rglob("*") if f.is_file()
              and "__pycache__" not in f.parts and f.suffix != ".pyc"]
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, "vangogh-lora/" + path.relative_to(ROOT).as_posix())
    with ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity test failed")
    print(f"{target}\n{target.stat().st_size} bytes\nsha256={hashlib.sha256(target.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
