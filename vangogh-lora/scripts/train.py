"""Launch the vendored official SDXL LoRA trainer with a recorded configuration."""
import argparse
import ast
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "vendor/train_text_to_image_lora_sdxl.py"


def command_for(config):
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    supported = {v.value[2:] for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                 for v in node.args if isinstance(v, ast.Constant) and isinstance(v.value, str)
                 and v.value.startswith("--")}
    unknown = set(config) - supported
    if unknown:
        raise ValueError(f"Arguments not supported by vendored trainer: {sorted(unknown)}")
    cmd = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes=1", "--num_machines=1",
           "--mixed_precision=" + config["mixed_precision"], "--dynamo_backend=no", str(TRAINER)]
    for name, value in config.items():
        if value is True:
            cmd.append("--" + name)
        elif value is not False and value is not None:
            cmd.extend(["--" + name, str(value)])
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.json")
    parser.add_argument("--smoke", action="store_true", help="20 steps, separate output, checkpoints every 10 steps")
    parser.add_argument("--resume", help="latest, or a checkpoint directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", help="Optional local model directory or Hub model id")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config.update(max_train_steps=20, checkpointing_steps=10,
                      dataloader_num_workers=0, output_dir="outputs/smoke")
    if args.max_steps is not None:
        if args.max_steps < 1:
            parser.error("max-steps must be positive")
        config["max_train_steps"] = args.max_steps
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.model:
        config["pretrained_model_name_or_path"] = args.model
    for key in ("train_data_dir", "output_dir"):
        path = Path(config[key])
        config[key] = str(path if path.is_absolute() else ROOT / path)
    data = Path(config["train_data_dir"])
    rows = [json.loads(s) for s in (data / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows or any(not r.get("text", "").strip() or not (data / r["file_name"]).is_file() for r in rows):
        raise ValueError("Missing images or empty training captions")
    if args.resume:
        config["resume_from_checkpoint"] = args.resume
    output = Path(config["output_dir"])
    if args.resume:
        candidates = [p for p in output.glob("checkpoint-*") if p.is_dir() and p.name[11:].isdigit()]
        if args.resume == "latest":
            if not candidates:
                raise ValueError("No checkpoint to resume; refusing to silently start a fresh run")
            checkpoint = max(candidates, key=lambda p: int(p.name[11:]))
        else:
            checkpoint = output / Path(args.resume).name
            if not checkpoint.is_dir():
                raise ValueError(f"Checkpoint must exist inside the output directory: {checkpoint}")
        if not (checkpoint / "pytorch_lora_weights.safetensors").is_file() or not list(checkpoint.glob("optimizer*")):
            raise ValueError(f"Checkpoint is incomplete: {checkpoint}")
    elif output.exists() and any(output.iterdir()) and not args.dry_run:
        raise ValueError("Output is not empty; use --resume latest or a new --output-dir")
    cmd = command_for(config)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    # Resolve remote weights to an immutable revision for this run; retain it on resume.
    previous = output / "run_config.json"
    if args.resume and previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old["pretrained_model_name_or_path"] != config["pretrained_model_name_or_path"]:
            raise ValueError("Cannot resume with a different base model")
        for key in ("resolution", "rank", "train_batch_size", "gradient_accumulation_steps", "mixed_precision"):
            if old[key] != config[key]:
                raise ValueError(f"Cannot resume after changing {key}")
        if old.get("revision"):
            config["revision"] = old["revision"]
        digest_file = output / "training_data_fingerprint.txt"
        digest = hashlib.sha256((data / "metadata.jsonl").read_bytes()).hexdigest()
        if digest_file.exists() and digest_file.read_text(encoding="utf-8").strip() != digest:
            raise ValueError("Training metadata changed; start a new run instead of resuming")
    if not Path(config["pretrained_model_name_or_path"]).is_dir() and not config.get("revision"):
        from huggingface_hub import model_info
        config["revision"] = model_info(config["pretrained_model_name_or_path"]).sha
    cmd = command_for(config)
    output.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metadata_bytes = (data / "metadata.jsonl").read_bytes()
    (output / "training_data_fingerprint.txt").write_text(hashlib.sha256(metadata_bytes).hexdigest(), encoding="utf-8")
    (output / f"captions-{stamp}.jsonl").write_bytes(metadata_bytes)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (output / f"environment-{stamp}.txt").write_text(freeze.stdout, encoding="utf-8")
    start = time.time()
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    code = -1
    try:
        with (output / f"train-{stamp}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
    finally:
        report = {"exit_code": code, "wall_seconds": time.time() - start, "config": config,
                  "caption_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                  "trainer_sha256": hashlib.sha256(TRAINER.read_bytes()).hexdigest(), "command": cmd}
        (output / f"run-{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
