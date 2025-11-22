import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-32B"
PEFT_PATH = "/home/ubuntu/tinker-cookbook/checkpoint"       # your LoRA dir
OUT_DIR   = "/home/ubuntu/tinker-cookbook/merged_model"     # where to save merged model

def main():
    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",   # or "cuda" if single big GPU, or "cpu" (needs huge RAM)
    )

    print(f"Loading LoRA adapter from: {PEFT_PATH}")
    model = PeftModel.from_pretrained(
        model,
        PEFT_PATH,
    )

    print("Merging LoRA into base weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {OUT_DIR}")
    model.save_pretrained(OUT_DIR)

    # optional: save tokenizer as well
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(OUT_DIR)

    print("Done.")

if __name__ == "__main__":
    main()
