from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import (
    SFTTrainer,  # simpler SFT loop  [oai_citation:9‡discuss.huggingface.co](https://discuss.huggingface.co/t/when-to-use-sfttrainer/40998)
)

MODEL_ID = "ibm-granite/granite-3.3-8b-instruct"  # base model 8 B  [oai_citation:10‡huggingface.co](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct)
TRAIN_CSV = "data/train.csv"
EVAL_CSV = "data/test.csv"
SEQ_LEN = 1024

# ---- 1.  load in 4-bit (QLoRA)  ----
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype="bfloat16"
)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", quantization_config=bnb_cfg
)
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# ---- 2.  attach LoRA adapters  ----
lora_cfg = LoraConfig(
    r=64, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()  # sanity-check

# ---- 3.  load tiny dataset  ----
ds = load_dataset(
    "csv",
    data_files={"train": TRAIN_CSV, "eval": EVAL_CSV},
    column_names=["system", "user", "assistant"],
)


# Helper to turn row → single prompt+completion
def fmt(ex):
    return {"prompt": f"{ex['system']}\n{ex['user']}", "completion": ex["assistant"]}


ds = ds.map(fmt, remove_columns=ds["train"].column_names)

# ---- 4.  trainer  ----
trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds["train"],
    eval_dataset=ds["eval"],
    max_seq_length=SEQ_LEN,
    deepspeed="ds_config_zero3.json",
    args=dict(
        per_device_train_batch_size=4,
        num_train_epochs=5,
        learning_rate=2e-4,
        logging_steps=5,
        output_dir="/checkpts/granite3.3-lora",
    ),
)
trainer.train()
trainer.save_model()

# (optional) merge adapters for inference-only use
# model = model.merge_and_unload()
# model.save_pretrained("/checkpts/granite3.3-lora-merged")
