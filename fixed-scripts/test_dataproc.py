import argparse
import logging
import os
import time

from datasets import load_dataset
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Helper function to reconstruct text similar to FunctionCallingDataset ---
# This is a simplified version of the original class's methods for direct use in .map()


def _safe_process_message_for_map(message_item):
    """Processes a single message item (dict, str, list) for map function."""
    if isinstance(message_item, dict):
        role = str(message_item.get("role", "")).lower()
        content = str(message_item.get("content", ""))
        if role == "system":
            return f"<|system|>\n{content}\n"
        elif role == "user":
            return f"<|user|>\n{content}\n"
        elif role == "assistant":
            return f"<|assistant|>\n{content}\n"
        else:
            return f"<|user|>\n{content}\n"  # Default to user
    elif isinstance(message_item, str):
        return f"<|user|>\n{message_item}\n"
    elif isinstance(message_item, list):
        result = ""
        for sub_message in message_item:
            result += _safe_process_message_for_map(sub_message)
        return result
    return f"<|user|>\n{str(message_item)}\n"


def _flush_content_for_map(
    current_role,
    current_content_list,
    formatted_text,
    next_role=None,
    next_content_line=None,
    force_flush=False,
):
    if current_role and current_content_list:
        role_content = "\n".join(current_content_list).strip()
        if role_content:
            if current_role == "system":
                if not formatted_text.startswith("<|system|>"):  # Basic check
                    formatted_text += f"<|system|>\n{role_content}\n"
            elif current_role == "user":
                formatted_text += f"<|user|>\n{role_content}\n"
            elif current_role == "assistant":
                formatted_text += f"<|assistant|>\n{role_content}\n"

    if force_flush:
        return None, [], formatted_text

    new_role = next_role
    new_content_list = [next_content_line.strip()] if next_content_line else []
    return new_role, new_content_list, formatted_text


def _create_labels(input_ids_list, tokenizer):
    """
    Creates labels for language modeling, masking non-assistant parts.
    input_ids_list: A list of token IDs.
    tokenizer: The tokenizer instance.
    """
    labels = list(input_ids_list)  # Make a mutable copy

    # Attempt to find assistant markers robustly
    # This is tricky because special tokens can be tokenized into multiple IDs or be part of vocab
    # The original script's method of decoding pairs is heuristic.
    # A truly robust method would involve finding the exact token ID sequence for "<|assistant|>" etc.
    # For now, we stick to a similar heuristic as the fine-tuning script for consistency.

    assistant_marker_str = "<|assistant|>"
    user_marker_str = "<|user|>"
    system_marker_str = "<|system|>"

    assistant_positions = []

    # This simplified search might miss markers if they are split by the tokenizer in a way
    # that `decode` on small windows doesn't reconstruct them.
    # The original fine-tuning script decodes slices of 2 tokens.
    # Let's try to emulate that, but be aware of its limitations.
    for i in range(len(input_ids_list) - 1):  # Iterate up to the second to last token
        # Decode a small window of tokens.
        # Using skip_special_tokens=False to ensure markers are decoded if they are special.
        # However, if they are not registered special tokens, this flag has no effect on them.
        try:
            # It's important that tokenizer.decode can handle arbitrary slices.
            # Some tokenizers might expect complete sequences or valid UTF-8.
            decoded_slice = tokenizer.decode(
                input_ids_list[i : i + 2], skip_special_tokens=False
            )
        except:  # Broad except as tokenizer internals can vary
            decoded_slice = ""

        if assistant_marker_str in decoded_slice:
            # This position 'i' is where the assistant marker *starts* or is detected within this window.
            assistant_positions.append(i)

    if assistant_positions:
        in_assistant_segment = False
        # Mask tokens up to the first assistant token, and the assistant token itself.
        # Then, unmask tokens until a new user/system token or end of sequence.

        # More direct approach: iterate tokens, switch state, mask accordingly.
        current_pos = 0
        while current_pos < len(labels):
            is_assistant_start = False
            # Check if current_pos is one of the identified assistant_positions
            # This check needs to be robust to multi-token markers.
            # The original logic was: if i in assistant_positions: labels[i:i+2] = -100
            # This implies the marker is roughly 2 tokens.

            # Let's try to find the *actual* assistant marker tokens and mask them.
            # This is still heuristic without knowing the exact token IDs of markers.

            # Simplified logic based on the original script's intent:
            # Mask everything that is not explicitly part of an assistant's response.

            temp_in_assistant = False
            for i in range(len(labels)):
                # Try to detect start of assistant turn
                if i in assistant_positions:  # Heuristic: marker starts at i
                    temp_in_assistant = True
                    # Mask the marker itself (assuming 2 tokens for simplicity like original)
                    labels[i] = -100
                    if i + 1 < len(labels):
                        labels[i + 1] = -100
                    continue  # Move to token after marker

                if not temp_in_assistant:
                    labels[i] = -100
                else:  # We are in an assistant turn
                    # Check for end of assistant turn (start of user/system)
                    # This also needs robust detection of user/system markers
                    is_user_or_system_marker = False
                    if i + 1 < len(labels):
                        try:
                            decoded_slice_end = tokenizer.decode(
                                input_ids_list[i : i + 2], skip_special_tokens=False
                            )
                        except:
                            decoded_slice_end = ""
                        if (
                            user_marker_str in decoded_slice_end
                            or system_marker_str in decoded_slice_end
                        ):
                            is_user_or_system_marker = True

                    if is_user_or_system_marker:
                        temp_in_assistant = False
                        labels[i] = -100  # Mask the user/system marker
                        if i + 1 < len(labels):
                            labels[i + 1] = -100
            break  # Broke out of the while current_pos loop, used temp_in_assistant logic instead

    else:  # No assistant token found
        for i in range(len(labels)):
            labels[i] = -100

    return labels


def preprocess_example(example, tokenizer, max_length):
    """
    Processes a single raw example from the dataset into tokenized form
    with input_ids, attention_mask, and labels.
    """
    idx = example.get("_idx_internal", None)  # If we add an index during enumeration

    formatted_text = ""
    # System message
    if "system" in example and example["system"] is not None:
        system_content = str(example["system"]).strip()
        if system_content:
            formatted_text += f"<|system|>\n{system_content}\n"

    # Chat content
    chat_content = example.get("chat", example.get("text", str(example)))

    if isinstance(chat_content, str):
        lines = chat_content.split("\n")
        current_role = None
        current_content_list = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            new_role_detected = False
            if line.startswith("SYSTEM:"):
                current_role, current_content_list, formatted_text = (
                    _flush_content_for_map(
                        current_role,
                        current_content_list,
                        formatted_text,
                        "system",
                        line.replace("SYSTEM:", "").strip(),
                    )
                )
                new_role_detected = True
            elif line.startswith("USER:"):
                current_role, current_content_list, formatted_text = (
                    _flush_content_for_map(
                        current_role,
                        current_content_list,
                        formatted_text,
                        "user",
                        line.replace("USER:", "").strip(),
                    )
                )
                new_role_detected = True
            elif line.startswith("A:"):  # Assuming 'A:' is Assistant
                current_role, current_content_list, formatted_text = (
                    _flush_content_for_map(
                        current_role,
                        current_content_list,
                        formatted_text,
                        "assistant",
                        line.replace("A:", "").strip(),
                    )
                )
                new_role_detected = True
            elif line.startswith("FUNCTION RESPONSE:"):
                if current_role == "assistant":
                    current_content_list.append(line)
                else:
                    current_role, current_content_list, formatted_text = (
                        _flush_content_for_map(
                            current_role,
                            current_content_list,
                            formatted_text,
                            "assistant",
                            line,
                        )
                    )
                new_role_detected = True  # Treat as boundary

            if not new_role_detected:
                if current_role:
                    current_content_list.append(line)
                else:  # Default if no role seen yet
                    current_role = "system"
                    current_content_list = [line]

        # Flush any remaining content
        _, _, formatted_text = _flush_content_for_map(
            current_role, current_content_list, formatted_text, force_flush=True
        )

    elif isinstance(chat_content, list):
        for message_item in chat_content:
            formatted_text += _safe_process_message_for_map(message_item)
    else:
        formatted_text += f"<|user|>\n{str(chat_content)}\n"

    formatted_text = formatted_text.replace("<|endoftext|>", "")
    if not formatted_text.strip():
        formatted_text = (
            "<|user|>\nHello\n<|assistant|>\nHello! How can I help you today?\n"
        )

    # Tokenize
    # Ensure pad_token is set on the tokenizer instance passed to this function
    # tokenizer.pad_token = tokenizer.eos_token (should be done outside, once)
    encodings = tokenizer(
        formatted_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",  # Pad to max_length for consistent tensor shapes
        return_attention_mask=True,
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # Create labels
    labels = _create_labels(input_ids, tokenizer)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess FunctionCallingDataset and save to disk."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ibm-granite/granite-3.3-2b-instruct",
        help="Tokenizer model name or path.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="glaiveai/glaive-function-calling-v2",
        help="Name of the dataset on Hugging Face Hub.",
    )
    parser.add_argument(
        "--dataset_data_file",
        type=str,
        default="glaive-function-calling-v2.json",  # Confirmed by user
        help="Specific data file within the dataset (e.g., json, jsonl).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the processed dataset.",
    )
    parser.add_argument(
        "--num_samples_to_process",
        type=int,
        default=-1,
        help="Number of samples to process and save. -1 for all samples.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,  # os.cpu_count(),
        help="Number of processes to use for .map(). Defaults to all available CPUs.",
    )
    script_args = parser.parse_args()

    if script_args.num_proc is None:
        script_args.num_proc = os.cpu_count()
        logger.info(f"Using {script_args.num_proc} processes for dataset mapping.")

    logger.info(f"Loading tokenizer: {script_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}"
        )

    logger.info(f"Loading raw dataset: {script_args.dataset_name}")
    try:
        # Load the specific data file for the 'train' split
        raw_dataset_dict = load_dataset(
            script_args.dataset_name,
            data_files={"train": script_args.dataset_data_file},
        )
        if "train" not in raw_dataset_dict:
            logger.error(
                f"'train' split not found in loaded dataset. Available splits: {list(raw_dataset_dict.keys())}"
            )
            exit(1)
        raw_dataset_train = raw_dataset_dict["train"]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    logger.info(f"Raw 'train' dataset loaded. Total examples: {len(raw_dataset_train)}")

    if (
        script_args.num_samples_to_process > 0
        and script_args.num_samples_to_process < len(raw_dataset_train)
    ):
        raw_dataset_subset = raw_dataset_train.select(
            range(script_args.num_samples_to_process)
        )
        logger.info(f"Processing a subset of {len(raw_dataset_subset)} samples.")
    else:
        raw_dataset_subset = raw_dataset_train
        logger.info(
            f"Processing all {len(raw_dataset_subset)} samples from 'train' split."
        )

    # Prepare for mapping
    # The `preprocess_example` function needs `tokenizer` and `max_length`.
    # We can pass these using `fn_kwargs` in the `.map()` call.

    logger.info(
        f"Starting dataset preprocessing with {script_args.num_proc} processes..."
    )
    start_map_time = time.time()

    processed_dataset = raw_dataset_subset.map(
        preprocess_example,
        fn_kwargs={"tokenizer": tokenizer, "max_length": script_args.max_seq_length},
        num_proc=script_args.num_proc,
        remove_columns=raw_dataset_subset.column_names,  # Remove old columns to keep only processed ones
    )

    map_duration = time.time() - start_map_time
    logger.info(f"Dataset preprocessing finished in {map_duration:.2f} seconds.")

    # Set format to PyTorch tensors for easier loading in the training script
    # processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    # Saving to disk in Arrow format is generally fine; set_format can be done after loading.

    logger.info(f"Saving processed dataset to {script_args.output_path}...")
    start_save_time = time.time()
    processed_dataset.save_to_disk(script_args.output_path)
    save_duration = time.time() - start_save_time
    logger.info(f"Processed dataset saved in {save_duration:.2f} seconds.")
    logger.info("--- Preprocessing Complete ---")
