"""SDXL image-to-image with optional LoRA and a seed-matched base-model comparison."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def fit_image(image, max_side):
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = max_side / max(image.size)
    width, height = [max(8, round(s * scale / 8) * 8) for s in image.size]
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="One image or a directory")
    p.add_argument("--output", type=Path, required=True, help="New directory, never overwritten")
    p.add_argument("--lora", type=Path, help="Final training output or checkpoint directory")
    p.add_argument("--compare-base", action="store_true")
    p.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--revision", help="Base model commit; inferred from LoRA run_config.json when available")
    p.add_argument("--prompt", default="a painting in the style of Vincent van Gogh")
    p.add_argument("--negative-prompt", default="text, watermark, blurry, distorted")
    p.add_argument("--strength", type=float, default=0.5)
    p.add_argument("--lora-scale", type=float, default=0.8)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--limit", type=int, default=0, help="0 means all images")
    p.add_argument("--cpu-offload", action="store_true")
    args = p.parse_args()
    if not 0 < args.strength <= 1 or args.steps < 1 or int(args.steps * args.strength) < 1:
        p.error("strength must be in (0, 1], and steps * strength must be >= 1")
    if args.max_side < 64 or args.max_side % 8 or args.limit < 0 or args.lora_scale < 0:
        p.error("max-side must be >=64 and divisible by 8; limit/LoRA scale must be nonnegative")
    if args.compare_base and not args.lora:
        p.error("--compare-base requires --lora")
    if args.lora and not (args.lora / "pytorch_lora_weights.safetensors").is_file():
        p.error("LoRA directory must contain pytorch_lora_weights.safetensors")
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = sorted(f for f in args.input.iterdir() if f.suffix.lower() in extensions) if args.input.is_dir() else [args.input]
    if not images or any(not f.is_file() for f in images):
        p.error("No valid input images")
    if args.limit:
        images = images[:args.limit]
    if args.output.exists():
        p.error("Output already exists; choose a new output directory")
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline
    if not torch.cuda.is_available():
        p.error("CUDA GPU required for this first-version pipeline")
    if args.lora:
        for path in [args.lora / "run_config.json", args.lora.parent / "run_config.json"]:
            if path.exists():
                config = json.loads(path.read_text(encoding="utf-8"))
                if config["pretrained_model_name_or_path"] != args.model:
                    p.error("Use the same --model as training")
                if args.revision and config.get("revision") and args.revision != config["revision"]:
                    p.error("Use the same --revision as training")
                args.revision = args.revision or config.get("revision")
                break
    if not Path(args.model).is_dir() and not args.revision:
        from huggingface_hub import model_info
        args.revision = model_info(args.model).sha
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16, use_safetensors=True)
    if args.lora:
        pipe.load_lora_weights(str(args.lora), adapter_name="vangogh")
        pipe.set_adapters("vangogh", adapter_weights=args.lora_scale)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.enable_vae_tiling()
    args.output.mkdir(parents=True)
    (args.output / "settings.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as results:
        for i, path in enumerate(images):
            with Image.open(path) as source:
                init = fit_image(source, args.max_side)
            seed = args.seed + i
            panels = [("Input (resized)", init)]
            modes = ["base", "lora"] if args.compare_base else ["lora" if args.lora else "base"]
            record = {"input": str(path), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "seed": seed, "size": init.size, "outputs": {}}
            stem = f"{i:04d}_{path.stem}"
            for mode in modes:
                if args.lora:
                    pipe.disable_lora() if mode == "base" else pipe.enable_lora()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    image = pipe(prompt=args.prompt, negative_prompt=args.negative_prompt, image=init,
                                 strength=args.strength, num_inference_steps=args.steps,
                                 guidance_scale=args.guidance, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
                torch.cuda.synchronize()
                filename = f"{stem}_{mode}.png"
                image.save(args.output / filename)
                record["outputs"][mode] = {"file": filename, "seconds": time.perf_counter() - start,
                                            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
                panels.append((mode, image))
            width, height = init.size
            grid = Image.new("RGB", (width * len(panels), height + 28), "white")
            draw = ImageDraw.Draw(grid)
            for column, (label, image) in enumerate(panels):
                grid.paste(image, (column * width, 28))
                draw.text((column * width + 8, 7), label, fill="black")
            grid.save(args.output / f"{stem}_comparison.png")
            results.write(json.dumps(record) + "\n")
            results.flush()
            print(f"{i + 1}/{len(images)}: {stem}", flush=True)


if __name__ == "__main__":
    main()
