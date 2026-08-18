from __future__ import annotations

import json
import platform
import sys

import torch

from double_attention.triton_kernels import TRITON_AVAILABLE


def main() -> None:
    cuda_available = torch.cuda.is_available()
    report: dict[str, object] = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton_available": TRITON_AVAILABLE,
        "cuda_available": cuda_available,
    }
    if cuda_available:
        device = torch.device("cuda")
        report.update(
            {
                "device": torch.cuda.get_device_name(device),
                "capability": torch.cuda.get_device_capability(device),
                "bf16": torch.cuda.is_bf16_supported(),
            }
        )
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        left = torch.randn(512, 512, device=device, dtype=dtype)
        result = left @ left
        torch.cuda.synchronize(device)
        report["cuda_gemm_finite"] = bool(torch.isfinite(result).all())

    print(json.dumps(report, indent=2))
    if not cuda_available:
        raise SystemExit("CUDA is unavailable")
    if not TRITON_AVAILABLE:
        raise SystemExit("Triton is unavailable")


if __name__ == "__main__":
    main()
