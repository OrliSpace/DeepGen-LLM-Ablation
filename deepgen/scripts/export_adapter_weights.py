import os
import argparse
import torch

def export_adapter_weights(
    checkpoint_path="work_dirs/sft_llm_ablation/iter_50000.pth",
    output_path="work_dirs/deepgen_sft_llm_adapter_50k.pt"
):
    print(f"Loading full training checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = raw_ckpt.get("state_dict", raw_ckpt)
    
    adapter_state_dict = {}
    for key, tensor in state_dict.items():
        clean_key = key.replace("module.", "").replace("model.", "")
        if "adapter" in clean_key:
            adapter_state_dict[clean_key] = tensor.clone().detach().cpu()
            print(f"  Extracted: {clean_key} | Shape: {list(tensor.shape)} | dtype: {tensor.dtype}")
            
    print(f"\nTotal adapter parameters extracted: {len(adapter_state_dict)}")
    assert len(adapter_state_dict) > 0, "No adapter parameters found in checkpoint!"
    
    # Verify L1 norm
    total_l1_norm = sum(p.abs().sum().item() for p in adapter_state_dict.values())
    total_params = sum(p.numel() for p in adapter_state_dict.values())
    
    print(f"Total Adapter Parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"Total Adapter L1 Weight Norm: {total_l1_norm:.4f}")
    assert total_l1_norm > 100.0, f"Adapter L1 norm is too low ({total_l1_norm:.4f}), weights appear uninitialized!"
    
    # Save standalone checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_payload = {
        "state_dict": adapter_state_dict,
        "metadata": {
            "model_name": "DeepGenSFTLLMAdapter",
            "iteration": 50000,
            "total_params": total_params,
            "l1_norm": total_l1_norm,
            "description": "Standalone zero-initialized residual LLM adapter weights for DeepGen 1.0"
        }
    }
    torch.save(save_payload, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSuccessfully exported standalone adapter weights to: {output_path} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export standalone adapter weights from training checkpoint.")
    parser.add_argument("--checkpoint", default="work_dirs/sft_llm_ablation/iter_50000.pth", help="Path to full checkpoint")
    parser.add_argument("--output", default="work_dirs/deepgen_sft_llm_adapter_50k.pt", help="Path to output standalone .pt file")
    args = parser.parse_args()
    export_adapter_weights(args.checkpoint, args.output)
