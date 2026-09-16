"""CPU tests for data leakage and the launch contract; no model download."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class WorkflowTests(unittest.TestCase):
    def test_duplicate_images_never_cross_prepared_splits(self):
        prepare = module("prepare_data").prepare
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "input.zip"
            with ZipFile(archive, "w") as z:
                for split, values in {"trainA": [1, 2, 3, 4, 4], "testA": [1, 2, 5],
                                      "trainB": [10, 11, 12, 13], "testB": [10, 14]}.items():
                    for i, value in enumerate(values):
                        buf = io.BytesIO()
                        Image.new("RGB", (16, 16), (value, value, value)).save(buf, format="PNG")
                        z.writestr(f"dataset/{split}/{i}.png", buf.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                report = prepare(archive, root / "out", holdout=1, eval_count=2)
                prepare(archive, root / "out2", holdout=1, eval_count=2)
            records = json.loads((root / "out/manifest.json").read_text())
            hashes = lambda split: {r["pixel_sha256"] for r in records if r["destination"].startswith(split + "/")}
            self.assertFalse(hashes("train") & hashes("val_style"))
            self.assertFalse(hashes("eval_photos") & hashes("test_photos"))
            self.assertFalse((hashes("train") | hashes("val_style")) & hashes("test_style_independent"))
            self.assertEqual(report["prepared_counts"]["train"], 3)
            self.assertEqual(report["train_test_pixel_overlap"], {"A": 2, "B": 1})
            self.assertEqual((root / "out/manifest.json").read_bytes(), (root / "out2/manifest.json").read_bytes())
            with self.assertRaises(ValueError):
                prepare(archive, root / "out", holdout=1, eval_count=2)

    def test_config_flags_are_supported_by_actual_vendor_script(self):
        train = module("train")
        config = json.loads((ROOT / "configs/train.json").read_text())
        cmd = train.command_for(config)
        self.assertIn("--gradient_checkpointing", cmd)
        self.assertNotIn("--train_text_encoder", cmd)
        with self.assertRaises(ValueError):
            train.command_for({**config, "invented_training_flag": True})

    def test_inference_dimensions_preserve_aspect_with_bounded_rounding(self):
        infer = module("infer")
        output = infer.fit_image(Image.new("RGB", (300, 200)), 512)
        self.assertEqual(output.width, 512)
        self.assertEqual(output.height % 8, 0)
        self.assertLess(abs(output.width / output.height - 1.5), 0.03)


if __name__ == "__main__":
    unittest.main()
