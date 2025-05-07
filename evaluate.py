import evaluate
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from peft import PeftModel
import argparse
import json
import os
import mlflow


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned LLM with LoRA adapters."
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        required=True,
        help="Base model ID from Hugging Face Hub.",
    )
    parser.add_argument(
        "--adapter_load_path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapters (output_dir from training).",
    )
    parser.add_argument(
        "--test_csv_path",
        type=str,
        required=True,
        help="Path to the test CSV file (e.g., data/test_extended.csv).",
    )
    parser.add_argument(
        "--output_results_file",
        type=str,
        default="evaluation_results.json",
        help="File to save evaluation results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., 'cuda:0', 'cpu'). Defaults to cuda if available, else cpu.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens for generation pipeline.",
    )
    parser.add_argument(
        "--no_mlflow",
        action="store_true",
        help="Disable MLflow logging for this script.",
    )
    cli_args = parser.parse_args()

    if cli_args.device:
        device_map = {"": cli_args.device}  # For pipeline device placement
        torch_device = torch.device(cli_args.device)
    else:
        if torch.cuda.is_available():
            device_map = {
                "": "cuda:0"
            }  # Default to first GPU for pipeline if not specified
            torch_device = torch.device("cuda:0")
        else:
            device_map = {"": "cpu"}
            torch_device = torch.device("cpu")
    print(f"Evaluation running on device: {torch_device}")

    # MLflow Setup (run this before model loading to catch potential OOMs as params)
    run_name = f"evaluate_{os.path.basename(cli_args.adapter_load_path)}_{os.path.basename(cli_args.test_csv_path)}"
    if not cli_args.no_mlflow:
        try:
            mlflow.start_run(run_name=run_name)
            print(f"MLflow run started for evaluation: {run_name}")
            mlflow.log_params(vars(cli_args))
        except Exception as e:
            print(
                f"WARN: MLflow start_run failed: {e}. Proceeding without MLflow for this script."
            )
            cli_args.no_mlflow = True  # Disable further mlflow calls if start fails

    # Load the base model (quantized or not, depending on how it was trained/saved)
    # For QLoRA, the adapters are applied to a quantized base model.
    # If loading adapters to a non-quantized model, ensure it's the same base architecture.
    print(f"Loading base model: {cli_args.base_model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Assuming adapters were trained on a 4-bit quantized model
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # Should match training compute_dtype
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        cli_args.base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,  # Match training compute_dtype for consistency
        # device_map="auto" # Let PEFT handle device placement after loading adapters or use specified device
    )
    tokenizer = AutoTokenizer.from_pretrained(cli_args.base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading LoRA adapters from: {cli_args.adapter_load_path}")
    # Load the PEFT model by applying adapters to the base model
    # Ensure the adapter_load_path points to the directory where adapter_model.safetensors etc. are saved
    model = PeftModel.from_pretrained(base_model, cli_args.adapter_load_path)
    model = model.to(torch_device)  # Move the composed model to the target device
    model.eval()  # Set to evaluation mode

    print("Creating text-generation pipeline...")
    # device=None will make pipeline use model.device if available, or device_map
    # For single GPU, model.to(device) is clearer.
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=torch_device,
        max_new_tokens=cli_args.max_new_tokens,
    )

    print(f"Loading test data from: {cli_args.test_csv_path}")
    em_metric = evaluate.load("exact_match")
    preds, refs = [], []
    prompts = []

    try:
        with open(cli_args.test_csv_path, "r") as f:
            for line_num, line in enumerate(f):
                try:
                    sys_p, usr_p, gold = line.rstrip("\n").split(",", 2)
                    # Using the same formatting as in finetune.py
                    prompt_text = f"<s>[INST] {sys_p}\n{usr_p} [/INST]"
                    prompts.append(prompt_text)
                    refs.append(gold)
                except ValueError:
                    print(
                        f"Skipping malformed line {line_num + 1} in {cli_args.test_csv_path}: {line.strip()}"
                    )
                    continue

        print(f"Generating predictions for {len(prompts)} prompts...")
        # Batch generation if pipeline supports it well and for efficiency, though simple loop is fine for clarity here
        generated_outputs = pipe(
            prompts, batch_size=4
        )  # Example batch_size for pipeline

        for output in generated_outputs:
            # The output of pipeline for text-generation is a list of dicts for each prompt
            # Each dict is like: {'generated_text': 'FULL_PROMPT_AND_RESPONSE'}
            # We need to extract only the generated part, after the prompt.
            full_text = output[0]["generated_text"]
            # Find the end of our prompt marker to extract only the assistant's response
            prompt_marker = "[/INST]"
            marker_index = full_text.rfind(prompt_marker)
            if marker_index != -1:
                prediction = full_text[marker_index + len(prompt_marker) :].strip()
            else:  # Fallback if marker not found (should not happen with this prompt format)
                prediction = full_text
            preds.append(prediction)

        results = {}
        if preds and refs:
            em_score = em_metric.compute(predictions=preds, references=refs)
            print(f"Exact Match Score: {em_score}")
            results["exact_match"] = em_score
            if not cli_args.no_mlflow and mlflow.active_run():
                mlflow.log_metrics(em_score)
        else:
            print("No valid predictions or references to compute metrics.")
            results["exact_match"] = None

        # Save detailed results
        output_data = {
            "cli_args": vars(cli_args),
            "metrics": results,
            "predictions": preds,
            "references": refs,
            "prompts": prompts,
        }
        with open(cli_args.output_results_file, "w") as f_out:
            json.dump(output_data, f_out, indent=2)
        print(f"Evaluation results saved to {cli_args.output_results_file}")
        if not cli_args.no_mlflow and mlflow.active_run():
            mlflow.log_artifact(cli_args.output_results_file)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        if not cli_args.no_mlflow and mlflow.active_run():
            mlflow.log_param("evaluation_error", str(e))
        raise
    finally:
        if not cli_args.no_mlflow and mlflow.active_run():
            mlflow.end_run()
            print("MLflow run ended for evaluation.")


if __name__ == "__main__":
    main()
