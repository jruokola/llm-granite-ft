#!/usr/bin/env python3
"""
Wrapper script to choose between local macOS finetuner or MLX CLI.
If a Granite-processed Arrow directory with `processed_data.jsonl` is provided,
it invokes the local `function-finetune-osx.py`. Otherwise, it uses MLX’s `mlx_lm.lora`.
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(__file__)
LOCAL_SCRIPT = os.path.join(SCRIPT_DIR, "function-finetune-osx.py")


def parse_args():
    parser = argparse.ArgumentParser(
        description="macOS LoRA/QLoRA finetuning: Granite or MLX"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ibm-granite/granite-3.3-2b-instruct",
        help="Model identifier or local path",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Arrow dataset dir or JSONL folder",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        help="Hugging Face dataset identifier",
    )
    parser.add_argument("--iters", type=int, default=600, help="Training iterations")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_layers", type=int, default=16, help="Layers to tune")
    parser.add_argument(
        "--adapter_path", type=str, default="./adapters", help="Adapter output"
    )
    parser.add_argument("--mask_prompt", action="store_true", help="Mask prompt")
    parser.add_argument(
        "--grad_checkpoint", action="store_true", help="Use gradient checkpointing"
    )
    parser.add_argument("--wandb", type=str, help="W&B project name")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
    )
    args = parse_args()
    # Determine raw data source
    source = args.hf_dataset or args.data_dir
    # Detect Arrow-format shards (Granite output)
    arrow_flag = os.path.isdir(source) and any(
        f.endswith(".arrow") for f in os.listdir(source)
    )
    # Detect Arrow-format shards in directory
    # If Arrow-format shards or processed_data.jsonl found, use local macOS script
    if arrow_flag or (
        os.path.isdir(source)
        and os.path.isfile(os.path.join(source, "processed_data.jsonl"))
    ):
        proc_jsonl = os.path.join(source, "processed_data.jsonl")
        cmd = [
            sys.executable,
            LOCAL_SCRIPT,
            "--processed_dataset_path",
            source,
            "--model_name_or_path",
            args.model_name_or_path,
            "--batch_size",
            str(args.batch_size),
            "--gradient_accumulation_steps",
            str(args.batch_size),
            "--learning_rate",
            "1e-5",
            "--num_epochs",
            "5",
            "--dataset_subset_size",
            "-1",
            "--save_steps",
            "500",
            "--eval_steps",
            "10",
            "--warmup_ratio",
            "0.1",
            "--output_dir",
            "./function_calling_output_osx",
        ]
        if args.mask_prompt:
            cmd.append("--gradient_checkpointing")
    else:
        # Fallback to MLX CLI
        # If HF identifier, MLX will load directly; if JSONL folder, MLX expects train.jsonl/valid.jsonl
        workdir = tempfile.mkdtemp(prefix="mlx_lora_")
        if os.path.isdir(source):
            # assume JSONL in folder
            os.system(f"cp {source}/train.jsonl {workdir}/")
            os.system(f"cp {source}/valid.jsonl {workdir}/")
            data_arg = workdir
        else:
            data_arg = source
        cmd = [
            "mlx_lm.lora",
            "--model",
            args.model_name_or_path,
            "--train",
            "--data",
            data_arg,
            "--iters",
            str(args.iters),
            "--batch-size",
            str(args.batch_size),
            "--num-layers",
            str(args.num_layers),
            "--adapter-path",
            args.adapter_path,
        ]
        if args.mask_prompt:
            cmd.append("--mask-prompt")
        if args.grad_checkpoint:
            cmd.append("--grad-checkpoint")
        if args.wandb:
            cmd.extend(["--wandb", args.wandb])

    logging.info("Running command:\n  %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logging.error("Finetuning failed (exit %d)", result.returncode)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
