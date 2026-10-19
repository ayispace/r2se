from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

BASE = Path("models/Qwen2.5-3B-Instruct")
ADAPTER = Path("outputs/sft_full_v100/final_adapter")
OUTPUT = Path("models/Qwen2.5-3B-R2-SFT-E5")

assert BASE.exists(), f"Base model missing: {BASE}"
assert (ADAPTER / "adapter_config.json").exists()
assert (ADAPTER / "adapter_model.safetensors").exists()

# 防止意外覆盖已有模型
if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise RuntimeError(f"Output directory is not empty: {OUTPUT}")

config = PeftConfig.from_pretrained(str(ADAPTER))
print("Adapter base:", config.base_model_name_or_path)
print("LoRA rank:", config.r)

assert "Qwen2.5-3B-Instruct" in str(
    config.base_model_name_or_path
), "Adapter 与基础模型不匹配"

print("Loading base model...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    str(BASE),
    torch_dtype=torch.float16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

print("Loading LoRA adapter...", flush=True)

model = PeftModel.from_pretrained(
    model,
    str(ADAPTER),
    is_trainable=False,
)

model.eval()

print("Merging LoRA...", flush=True)

merged = model.merge_and_unload(safe_merge=True)

print("Saving merged model...", flush=True)

OUTPUT.mkdir(parents=True, exist_ok=True)

merged.save_pretrained(
    str(OUTPUT),
    safe_serialization=True,
    max_shard_size="2GB",
)

tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER))
tokenizer.save_pretrained(str(OUTPUT))

assert (OUTPUT / "config.json").exists()
assert list(OUTPUT.glob("*.safetensors"))

print("MERGE_SUCCESS", flush=True)
print("MODEL_PATH:", OUTPUT.resolve(), flush=True)
