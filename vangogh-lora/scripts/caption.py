"""Produce content + style captions with BLIP; review the generated metadata."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared/train"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    path = args.data / "metadata.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(args.device).eval()
    for index, row in enumerate(rows):
        with Image.open(args.data / row["file_name"]) as image:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(args.device)
        with torch.inference_mode():
            result = model.generate(**inputs, max_new_tokens=60)
        content = processor.decode(result[0], skip_special_tokens=True)
        row["text"] = f"{content}, a painting in the style of Vincent van Gogh"
        print(f"{index + 1}/{len(rows)}: {row['text']}", flush=True)
    # Save separately: machine captions need review before replacing training metadata.
    target = args.data / "captions_blip.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"Review {target}, then copy it over metadata.jsonl. No training captions were overwritten.")


if __name__ == "__main__":
    main()
