import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from mmengine.config import Config
from xtuner.registry import BUILDER


def run_smoke_test(config_path="configs/models/deepgen_sft_llm_adapter.py", output_dir="outputs/smoke_test_eval"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== [Smoke Test] 1. Loading model configuration from: {config_path} ===")
    config = Config.fromfile(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("=== [Smoke Test] 2. Building DeepGenSFTLLMAdapter model ===")
    model = BUILDER.build(config.model)
    model = model.to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.eval()
    print("Model successfully built and initialized in eval mode.")

    # Generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(42)

    # -------------------------------------------------------------
    # Test 1: Text-to-Image (T2I) Forward Pass
    # -------------------------------------------------------------
    t2i_prompt = "A green apple next to a blue ceramic cup on a wooden table."
    print(f"\n=== [Smoke Test] 3. Running T2I Generation ===")
    print(f"Prompt: '{t2i_prompt}'")

    with torch.no_grad():
        t2i_images = model.generate(
            prompt=[t2i_prompt],
            cfg_prompt=[""],
            pixel_values_src=None,
            cfg_scale=4.0,
            num_steps=5,  # fast 5-step Euler flow matching for smoke test
            progress_bar=True,
            generator=generator,
            height=512,
            width=512,
        )

    # Model output is in [-1, 1] range from latents_to_pixels
    t2i_clamped = torch.clamp(127.5 * t2i_images + 128.0, 0, 255).to("cpu", dtype=torch.uint8)
    t2i_pil = Image.fromarray(t2i_clamped[0].permute(1, 2, 0).numpy())
    t2i_save_path = os.path.join(output_dir, "t2i_sample.png")
    t2i_pil.save(t2i_save_path)
    print(f"[Smoke Test] T2I Sample successfully generated and saved to: {t2i_save_path}")
    print(f"T2I Output Tensor Shape: {t2i_images.shape}, Pixel Range: [{t2i_images.min().item():.2f}, {t2i_images.max().item():.2f}]")

    # -------------------------------------------------------------
    # Test 2: Instruction-based Image Editing Forward Pass
    # -------------------------------------------------------------
    edit_prompt = "Change the apple to a red strawberry."
    print(f"\n=== [Smoke Test] 4. Running Instruction-Based Image Editing ===")
    print(f"Instruction: '{edit_prompt}'")

    # Prepare source image pixel tensor
    src_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    src_tensor = src_transform(t2i_pil).unsqueeze(0).to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)

    with torch.no_grad():
        edit_images = model.generate(
            prompt=[edit_prompt],
            cfg_prompt=["blurry, low quality, low resolution, distorted"],
            pixel_values_src=[src_tensor],
            cfg_scale=4.0,
            num_steps=5,
            progress_bar=True,
            generator=generator,
            height=512,
            width=512,
        )

    edit_clamped = torch.clamp(127.5 * edit_images + 128.0, 0, 255).to("cpu", dtype=torch.uint8)
    edit_pil = Image.fromarray(edit_clamped[0].permute(1, 2, 0).numpy())
    edit_save_path = os.path.join(output_dir, "edit_sample.png")
    edit_pil.save(edit_save_path)
    print(f"[Smoke Test] Image Editing Sample successfully generated and saved to: {edit_save_path}")
    print(f"Edit Output Tensor Shape: {edit_images.shape}, Pixel Range: [{edit_images.min().item():.2f}, {edit_images.max().item():.2f}]")

    # Save summary metadata
    summary = {
        "status": "PASSED",
        "t2i_prompt": t2i_prompt,
        "t2i_output": t2i_save_path,
        "edit_prompt": edit_prompt,
        "edit_output": edit_save_path,
        "tensor_shape": list(t2i_images.shape),
    }
    import json
    with open(os.path.join(output_dir, "smoke_test_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==================================================")
    print("✅ EVALUATION & INFERENCE SMOKE TEST PASSED 100%!")
    print("==================================================")
    return True


if __name__ == "__main__":
    run_smoke_test()
