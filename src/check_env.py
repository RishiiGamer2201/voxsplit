"""VoxSplit environment check.

Verifies the whole stack is importable and — critically — that the Blackwell
RTX 5070 Ti (sm_120) actually executes CUDA kernels, not just that CUDA is
"available". Run:  python src/check_env.py
"""
import platform
import sys


def main() -> int:
    print(f"Python      : {platform.python_version()}  ({sys.executable})")

    import torch

    print(f"torch       : {torch.__version__}")
    print(f"CUDA build  : {torch.version.cuda}")
    print(f"cuDNN       : {torch.backends.cudnn.version()}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA avail  : {cuda_ok}")

    if cuda_ok:
        dev = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU         : {dev}  (compute capability {cap[0]}.{cap[1]})")
        arch_list = torch.cuda.get_arch_list()
        print(f"arch_list   : {arch_list}")
        sm = f"sm_{cap[0]}{cap[1]}"
        print(f"kernels for : {sm} present -> {sm in arch_list}")

        # Real kernel execution test — this is what actually proves the wheel
        # ships Blackwell kernels. A mismatched wheel fails right here.
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        print(f"matmul test : OK  (result sum={c.sum().item():.2f}, "
              f"device={c.device})")
    else:
        print("!! CUDA not available — training will fall back to CPU (slow).")

    import torchaudio
    import librosa
    import soundfile
    import speechbrain
    import numpy
    import scipy
    print(f"torchaudio  : {torchaudio.__version__}")
    print(f"librosa     : {librosa.__version__}")
    print(f"soundfile   : {soundfile.__version__}")
    print(f"speechbrain : {speechbrain.__version__}")
    print(f"numpy       : {numpy.__version__}")
    print(f"scipy       : {scipy.__version__}")

    # ffmpeg availability (torchaudio uses it for many formats)
    import shutil
    print(f"ffmpeg      : {shutil.which('ffmpeg') or 'NOT FOUND'}")

    print("\nAll good — environment is ready." if cuda_ok else
          "\nEnv imports OK but no GPU acceleration.")
    return 0 if cuda_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
