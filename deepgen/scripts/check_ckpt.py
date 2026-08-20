import sys
import torch

print("Starting checkpoint verification...", flush=True)
ckpt_path = "work_dirs/sft_llm_ablation/iter_50000.pth"
print(f"Loading {ckpt_path}...", flush=True)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"Loaded successfully! Checkpoint keys: {list(ckpt.keys())}", flush=True)

state_dict = ckpt.get("state_dict", ckpt)
print(f"Total state dict entries: {len(state_dict)}", flush=True)

adapter_keys = [k for k in state_dict.keys() if "adapter" in k]
print(f"Found {len(adapter_keys)} adapter parameters:", flush=True)
for k in adapter_keys:
    tensor = state_dict[k]
    print(f"  {k}: shape={list(tensor.shape)}, dtype={tensor.dtype}, norm={tensor.float().norm().item():.6f}, mean={tensor.float().mean().item():.6f}, std={tensor.float().std().item():.6f}", flush=True)

# Also check other key weights if present
other_keys = [k for k in state_dict.keys() if "adapter" not in k]
print(f"Found {len(other_keys)} non-adapter keys in checkpoint (e.g. {other_keys[:5]})", flush=True)
print("VERIFICATION COMPLETE!", flush=True)
