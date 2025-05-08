from transformers import AutoTokenizer, AutoModelForCausalLM

AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-8b-instruct", cache_dir=".cache")
AutoModelForCausalLM.from_pretrained(
    "ibm-granite/granite-3.3-8b-instruct", cache_dir=".cache"
)
