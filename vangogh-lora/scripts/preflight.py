"""Check GPU support with actual BF16 attention and backward, without model download."""
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is unavailable. Install the cu128 build and check the host driver.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This configuration requires BF16 support.")
    q = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    F.scaled_dot_product_attention(q, q, q).float().square().mean().backward()
    torch.cuda.synchronize()
    names = ["torch", "torchvision", "diffusers", "transformers", "peft", "accelerate", "datasets"]
    # Import these together to catch dependency/API conflicts before allocating the full model.
    import torchvision, diffusers, transformers, peft, accelerate, datasets  # noqa: F401
    report = {"gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
              "torch_cuda": torch.version.cuda, "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
              "free_disk_gib": shutil.disk_usage(".").free / 2**30,
              "versions": {name: importlib.metadata.version(name) for name in names},
              "bf16_attention_backward": "passed"}
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()
