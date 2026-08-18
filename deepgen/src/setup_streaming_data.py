import os
import sys
import json

# Prevent name shadowing: Python incorrectly tries to load the local `src/datasets/` directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != script_dir]

from datasets import load_dataset

def main():
    print("Streaming dataset conceptual_captions...")
    dataset = load_dataset("conceptual_captions", split="train", streaming=True)
    
    os.makedirs("data", exist_ok=True)
    output_file = "data/t2i_pretrain_stream.json"
    
    samples = []
    count = 0
    
    for item in dataset:
        if count >= 50000:
            break
        
        # Accommodate OpenUni typical text/url column names
        prompt = item.get("caption", item.get("prompt", item.get("text", "")))
        url = item.get("url", item.get("image_url", item.get("image", "")))
        
        if prompt and url and isinstance(url, str) and url.startswith("http"):
            samples.append({
                "type": "T2I_SFT",
                "txt": prompt,
                "image": "",
                "image_path": url
            })
            count += 1
            if count % 5000 == 0:
                print(f"Processed {count} samples...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully saved {len(samples)} samples to {output_file}.")
    # Force exit to prevent pyarrow PyGILState_Release crashes during Python teardown
    os._exit(0)

if __name__ == "__main__":
    main()