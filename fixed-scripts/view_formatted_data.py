import argparse
import os  # Import os
import sys

from datasets import load_from_disk
from transformers import AutoTokenizer


def print_rank0(*args, **kwargs):
    # In case this script is ever run in a distributed-like env by mistake
    if int(os.getenv("RANK", "0")) == 0:
        print(*args, **kwargs)
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="View the first few examples of a processed dataset."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the processed dataset directory (saved by save_to_disk).",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="ibm-granite/granite-3.3-2b-instruct",
        help="Tokenizer model name or path to use for decoding.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=2,
        help="Number of examples to view.",
    )
    args = parser.parse_args()

    print_rank0(f"Loading tokenizer: {args.tokenizer_name_or_path}")
    # Ensure the tokenizer has any custom/special tokens if they were added during preprocessing/training
    # For viewing, this is less critical than for training, but good for accurate decoding.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path, trust_remote_code=True
    )
    # If pad token was added and not part of vocab, it might show as unk or special token string
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Add a common pad token if none exists, for decoding purposes
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print_rank0(f"Loading processed dataset from: {args.dataset_path}")
    try:
        processed_dataset = load_from_disk(args.dataset_path)
    except Exception as e:
        print_rank0(
            f"Failed to load dataset from {args.dataset_path}: {e}", file=sys.stderr
        )
        sys.exit(1)

    print_rank0(f"Dataset loaded. Total examples: {len(processed_dataset)}")
    print_rank0(f"Displaying first {args.num_examples} examples:\n")

    for i in range(min(args.num_examples, len(processed_dataset))):
        example = processed_dataset[i]

        print_rank0(f"--- Example {i + 1} ---")

        input_ids = example["input_ids"]
        # attention_mask = example["attention_mask"] # Not printed for brevity
        labels = example["labels"]

        decoded_text = tokenizer.decode(
            input_ids, skip_special_tokens=False
        )  # Show special tokens
        print_rank0("\n[DECODED TEXT]:")
        print_rank0(decoded_text)

        # Show where labels are active (not -100)
        unmasked_label_indices = [
            idx for idx, label_id in enumerate(labels) if label_id != -100
        ]
        if unmasked_label_indices:
            print_rank0("\n[LABELS (unmasked parts)]:")
            # For brevity, just show segments of unmasked labels
            current_segment_tokens = []
            current_segment_start_idx = -1

            for token_idx in range(len(input_ids)):
                if labels[token_idx] != -100:
                    if not current_segment_tokens:  # Start of a new segment
                        current_segment_start_idx = token_idx
                    current_segment_tokens.append(input_ids[token_idx])
                else:
                    if current_segment_tokens:  # End of a segment
                        decoded_segment = tokenizer.decode(
                            current_segment_tokens, skip_special_tokens=False
                        )
                        print_rank0(
                            f'  Tokens [{current_segment_start_idx}-{token_idx - 1}]: "{decoded_segment}"'
                        )
                        current_segment_tokens = []
                        current_segment_start_idx = -1

            if current_segment_tokens:  # Catch any trailing segment
                decoded_segment = tokenizer.decode(
                    current_segment_tokens, skip_special_tokens=False
                )
                print_rank0(
                    f'  Tokens [{current_segment_start_idx}-{len(input_ids) - 1}]: "{decoded_segment}"'
                )

        else:
            print_rank0("\n[LABELS]: All tokens are masked (-100).")

        print_rank0("-" * 20 + "\n")

    print_rank0("Done viewing examples.")
