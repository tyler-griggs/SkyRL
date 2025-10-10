import safetensors
from pathlib import Path
from huggingface_hub import snapshot_download

# For Qwen3-0.6B
print("Downloading Qwen3-0.6B...")
checkpoint_path_small = snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=["*.safetensors"])
keys_small = []
for file in Path(checkpoint_path_small).glob("*.safetensors"):
    print(f"Loading {file.name}...")
    # Just get the keys, not the actual tensors
    with safetensors.safe_open(file, framework="numpy") as f:
        keys_small.extend(f.keys())

# Print expert-related keys
print("\n=== Qwen3-0.6B Expert Keys ===")
expert_keys_small = [k for k in sorted(keys_small) if "expert" in k.lower()]
if expert_keys_small:
    for key in expert_keys_small[:20]:  # First 20
        print(key)
else:
    print("No expert keys found - this model may not have MoE layers")
    # Show first 10 keys to understand structure
    print("\nFirst 10 keys in checkpoint:")
    for key in sorted(keys_small)[:10]:
        print(key)

# For Qwen3-Coder-30B
print("\n\nDownloading Qwen3-Coder-30B-A3B-Instruct...")
checkpoint_path_large = snapshot_download("Qwen/Qwen3-Coder-30B-A3B-Instruct", allow_patterns=["*.safetensors"])
keys_large = []
for file in Path(checkpoint_path_large).glob("*.safetensors"):
    print(f"Loading {file.name}...")
    # Just get the keys, not the actual tensors
    with safetensors.safe_open(file, framework="numpy") as f:
        keys_large.extend(f.keys())

print("\n=== Qwen3-Coder-30B Expert Keys ===")
expert_keys_large = [k for k in sorted(keys_large) if "expert" in k.lower()]
if expert_keys_large:
    for key in expert_keys_large[:20]:
        print(key)
else:
    print("No expert keys found - this model may not have MoE layers")
    # Show first 10 keys to understand structure
    print("\nFirst 10 keys in checkpoint:")
    for key in sorted(keys_large)[:10]:
        print(key)