import torch


def get_default_device() -> torch.device:
    """Returns 'cuda' if available, otherwise 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_default_dtype(device: torch.device | None = None) -> torch.dtype:
    """
    Returns float16 for CUDA if supported, otherwise float32.
    """
    if device is None:
        device = get_default_device()

    if device.type == "cuda":
        # Check bfloat16 support (Ampere or newer, e.g. RTX 30xx+)
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def print_cuda_info() -> None:
    """Prints diagnostic information about CUDA and available GPUs."""
    available = torch.cuda.is_available()
    print(f"CUDA Available: {available}")
    if available:
        print(f"CUDA Version (PyTorch): {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"Device Count: {device_count}")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name} - Total Memory: {props.total_memory / (1024**2):.1f} MB")
            print(f"      Compute Capability: {props.major}.{props.minor}")
            print(f"      bfloat16 supported: {torch.cuda.is_bf16_supported()}")
