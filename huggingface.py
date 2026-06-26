import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, snapshot_download
import torch

model_name = "meta-llama/Llama-2-7b-hf"
hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

def print_token_help():
    print("If you still get 403, ensure your token has 'Enable access to public gated repositories' enabled:")
    print("  1) Visit https://huggingface.co/settings/tokens and edit the token's fine-grained settings.")
    print("  2) Or create a new token with the required permission.")
    print("To set token in shell (example):")
    print("  export HUGGINGFACE_HUB_TOKEN=hf_xxx")

try:
    model_id ="meta-llama/Llama-2-7b-hf"   # pick a realistic size first

    # Optional: speedier downloads with lower memory churn
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True, use_auth_token=hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,          # or torch.float16
        low_cpu_mem_usage=True,
        device_map="auto",                   # spreads layers to available GPU(s)/CPU
        offload_folder="/mnt/nvme/offload",  # make sure this exists & is fast
        trust_remote_code=True
    )

    # Inference example
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


except Exception as e:
    print("ERROR loading model:", str(e))
    print("Likely causes: insufficient RAM/GPU memory or gated repo access.")
    print("Suggestions:")
    print(" 1) Use a smaller model or run on a machine with more GPU memory.")
    print(" 2) Install bitsandbytes/accelerate and retry: pip install bitsandbytes accelerate safetensors")
    print(" 3) Create swap or increase memory if using CPU-only.")
    print(" 4) If you must use this model but can't load it, switch to source='openai' or download a local snapshot on a machine with enough RAM and point from_pretrained to the local path.")
    print("\nCheck system logs to confirm OOM:")
    print("  sudo dmesg | grep -i -E 'killed process|oom' || dmesg | tail -n 50")
    # re-raise if you want the script to stop
    raise