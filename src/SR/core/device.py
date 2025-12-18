import torch

class DeviceManager:
    def __init__(self, prefer_mps: bool = True):
        self.prefer_mps = prefer_mps

    def get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if self.prefer_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def recommended_precision(self, device: torch.device) -> dict:
        # HF Trainer flags
        if device.type == "cuda":
            # bf16 is often better on newer GPUs, but fp16 is safer default
            return {"fp16": True, "bf16": False}
        if device.type == "mps":
            # safest on Apple Silicon
            return {"fp16": False, "bf16": False}
        return {"fp16": False, "bf16": False}
