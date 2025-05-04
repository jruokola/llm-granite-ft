import evaluate
from transformers import pipeline

model_dir = "/checkpts/granite3.3-lora"  # or merged path
pipe = pipeline("text-generation", model=model_dir, max_new_tokens=256, device=0)

em = evaluate.load("exact_match")
preds, refs = [], []
for line in open("data/test.csv"):
    sys_p, usr_p, gold = line.rstrip("\n").split(",", 2)
    txt = f"{sys_p}\n{usr_p}"
    preds.append(pipe(txt)[0]["generated_text"])
    refs.append(gold)

print("Exact JSON match:", em.compute(predictions=preds, references=refs))
