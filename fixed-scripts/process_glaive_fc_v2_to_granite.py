import argparse
import json
import os
import shutil
import sys

from datasets import Dataset, DatasetInfo, Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer

# --- Granite Specific Tokens & Roles (consistent with generate_granite_fc_examples.py) ---
SOT = "<|start_of_role|>"
EOTR = "<|end_of_role|>"
EOTXT = "<|end_of_text|>"
TOOL_CALL_MARKER_GRANITE = "<|tool_call|>"

ROLE_SYSTEM_GRANITE = "system"
ROLE_AVAILABLE_TOOLS_GRANITE = "available_tools"
ROLE_USER_GRANITE = "user"
ROLE_ASSISTANT_GRANITE = "assistant"
ROLE_TOOL_RESPONSE_GRANITE = "tool_response"  # Matches 'function' role in glaive_fc_v2 when providing tool output


# --- Helper print functions ---
def log_info(*args, **kwargs):
    if int(os.getenv("RANK", "0")) == 0:
        __builtins__.print(*args, **kwargs)
        if "file" not in kwargs or kwargs["file"] == sys.stdout:
            sys.stdout.flush()


def log_error(*args, **kwargs):
    if int(os.getenv("RANK", "0")) == 0:
        __builtins__.print(*args, file=sys.stderr, **kwargs)
        sys.stderr.flush()


def log_warning(*args, **kwargs):
    if int(os.getenv("RANK", "0")) == 0:
        # Call the built-in print function, perhaps prefixing with [WARNING]
        # For consistency with log_error, printing to stderr.
        print_args = ("[WARNING]",) + args
        __builtins__.print(*print_args, file=sys.stderr, **kwargs)
        sys.stderr.flush()


# --- Formatting function (from generate_granite_fc_examples.py) ---
def format_granite_turn(role, content):
    content_str = str(content).strip()
    return f"{SOT}{role}{EOTR}{content_str}{EOTXT}\n"


# --- Labeling function (from generate_granite_fc_examples.py) ---
def create_labels_for_granite_sequence(input_ids_list, tokenizer):
    labels = [-100] * len(input_ids_list)
    decoded_full_text = tokenizer.decode(
        input_ids_list, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    token_offsets = []
    current_offset = 0
    for token_id in input_ids_list:
        decoded_token = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        token_length = len(decoded_token)
        token_offsets.append((current_offset, current_offset + token_length))
        current_offset += token_length

    assistant_turn_start_marker = f"{SOT}{ROLE_ASSISTANT_GRANITE}{EOTR}"
    search_start_char = 0
    while search_start_char < len(decoded_full_text):
        assistant_marker_char_start = decoded_full_text.find(
            assistant_turn_start_marker, search_start_char
        )
        if assistant_marker_char_start == -1:
            break
        unmask_content_char_start = assistant_marker_char_start + len(
            assistant_turn_start_marker
        )
        unmask_content_char_end = decoded_full_text.find(
            EOTXT, unmask_content_char_start
        )
        if unmask_content_char_end == -1:
            unmask_content_char_end = len(decoded_full_text)
        start_token_idx, end_token_idx = -1, -1
        for i, (tok_start_char, tok_end_char) in enumerate(token_offsets):
            if start_token_idx == -1 and tok_end_char > unmask_content_char_start:
                start_token_idx = i
            if start_token_idx != -1 and tok_start_char < unmask_content_char_end:
                end_token_idx = i
            if tok_start_char >= unmask_content_char_end and start_token_idx != -1:
                break
        if (
            start_token_idx != -1
            and end_token_idx != -1
            and end_token_idx >= start_token_idx
        ):
            is_padding_content = True
            for i_label in range(start_token_idx, end_token_idx + 1):
                if (
                    input_ids_list[i_label] != tokenizer.pad_token_id
                    and input_ids_list[i_label] != tokenizer.eos_token_id
                ):
                    is_padding_content = False
                    break
            if not is_padding_content:
                for i_label in range(start_token_idx, end_token_idx + 1):
                    if i_label < len(labels):
                        labels[i_label] = input_ids_list[i_label]
        search_start_char = unmask_content_char_end + len(EOTXT)
    return labels


# --- Main processing logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Download hqfx/glaive_fc_v2, format it for Granite, and save to disk."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the processed dataset.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="ibm-granite/granite-3.3-2b-instruct",
        help="Tokenizer model name or path.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,  # Defaulting to a more common length for function calling
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        default="hqfx/glaive_fc_v2",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--hf_dataset_split",
        type=str,
        default="train",
        help="Dataset split to process (e.g., 'train').",
    )
    script_args = parser.parse_args()

    # System prompt (same as in generate_granite_fc_examples.py)
    system_prompt = "Knowledge Cutoff Date: April 2024.\n Today's Date: April 12, 2025. You are Granite, developed by IBM. You are a helpful assistant with access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request."

    log_info(
        f"Loading source dataset: {script_args.hf_dataset_name}, split: {script_args.hf_dataset_split}"
    )
    try:
        source_dataset = load_dataset(
            script_args.hf_dataset_name, split=script_args.hf_dataset_split
        )
    except Exception as e:
        log_error(f"Failed to load dataset {script_args.hf_dataset_name}: {e}")
        sys.exit(1)

    log_info(f"Loaded {len(source_dataset)} examples from source.")

    formatted_text_examples = []
    for i, raw_example in enumerate(source_dataset):
        if i % 1000 == 0 and i > 0:
            log_info(f"Processing example {i}...")

        example_parts = []
        # 1. System Turn
        example_parts.append(format_granite_turn(ROLE_SYSTEM_GRANITE, system_prompt))

        # 2. Available Tools Turn
        # The 'functions' field in glaive_fc_v2 is already a JSON string of the tools list.
        # It might be a string representation of a list of tool JSONs.
        # The Granite format expects the content of ROLE_AVAILABLE_TOOLS_GRANITE to be the JSON string of the tools list.
        available_tools_content = raw_example.get(
            "functions", "[]"
        )  # Default to empty list if not present
        example_parts.append(
            format_granite_turn(ROLE_AVAILABLE_TOOLS_GRANITE, available_tools_content)
        )

        # 3. Conversation Turns
        conversation = raw_example.get("conversation", [])
        if not isinstance(conversation, list):
            log_error(
                f"Skipping example {i} due to invalid conversation format: {conversation}"
            )
            continue

        for turn in conversation:
            role = turn.get("role")
            content = turn.get("content")
            function_call_str = turn.get(
                "function_call"
            )  # This is a string in glaive_fc_v2

            if role == "user":
                example_parts.append(
                    format_granite_turn(ROLE_USER_GRANITE, content if content else "")
                )
            elif role == "assistant":
                if function_call_str:
                    # The function_call_str from glaive_fc_v2 is already the JSON string of the tool call list
                    # e.g., "[{\"arguments\": ..., \"name\": ...}]"
                    # Granite format: <|tool_call|>JSON_TOOL_CALL_LIST
                    assistant_response_content = (
                        f"{TOOL_CALL_MARKER_GRANITE}{function_call_str}"
                    )
                    example_parts.append(
                        format_granite_turn(
                            ROLE_ASSISTANT_GRANITE, assistant_response_content
                        )
                    )
                else:
                    example_parts.append(
                        format_granite_turn(
                            ROLE_ASSISTANT_GRANITE, content if content else ""
                        )
                    )
            elif (
                role == "function"
            ):  # This is ROLE_TOOL_RESPONSE_GRANITE in our target format
                # Content for 'function' role in glaive_fc_v2 is a string representation of a list containing a single JSON string.
                # e.g., "[\"{\\\"median\\\": 5}\"]"
                # We need to extract the inner JSON string.
                try:
                    # First, parse the outer list string:
                    tool_output_list_parsed = json.loads(content)
                    if (
                        isinstance(tool_output_list_parsed, list)
                        and len(tool_output_list_parsed) > 0
                    ):
                        actual_tool_output_str = tool_output_list_parsed[0]
                        example_parts.append(
                            format_granite_turn(
                                ROLE_TOOL_RESPONSE_GRANITE, actual_tool_output_str
                            )
                        )
                    else:
                        log_error(
                            f"Unexpected format for function role content in example {i}: {content}. Using raw content."
                        )
                        example_parts.append(
                            format_granite_turn(
                                ROLE_TOOL_RESPONSE_GRANITE, content if content else ""
                            )
                        )
                except json.JSONDecodeError:
                    log_error(
                        f"JSONDecodeError for function role content in example {i}: {content}. Using raw content."
                    )
                    example_parts.append(
                        format_granite_turn(
                            ROLE_TOOL_RESPONSE_GRANITE, content if content else ""
                        )
                    )
            else:
                log_warning(f"Unknown role '{role}' in example {i}. Skipping turn.")

        formatted_text_examples.append("".join(example_parts))

    log_info(f"Formatted {len(formatted_text_examples)} examples.")
    log_info(f"Loading tokenizer: {script_args.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name_or_path, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            log_info(
                f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}"
            )
        else:
            # Fallback if eos_token is also None (less common for instruct models)
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            log_info("Added [PAD] as pad_token.")
            # Important: If a new pad_token is added, the model's embeddings might need resizing
            # if this tokenizer is used for a model that didn't originally have it.
            # This script only processes data; model adjustments are for training scripts.

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    log_info("Tokenizing and creating labels for formatted examples...")
    for i, text_content in enumerate(formatted_text_examples):
        if i % 1000 == 0 and i > 0:
            log_info(f"Tokenizing example {i}...")
        if i < 2:  # Print first 2 formatted texts for verification
            log_info(f"\n--- Formatted Example {i + 1} Text (to be tokenized) ---")
            log_info(
                text_content[:1000] + "..."
                if len(text_content) > 1000
                else text_content
            )

        encodings = tokenizer(
            text_content,
            max_length=script_args.max_seq_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        labels = create_labels_for_granite_sequence(input_ids, tokenizer)

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    # Define features for the dataset (consistent with generate_granite_fc_examples.py)
    features = Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int32"), length=-1),
            "attention_mask": Sequence(feature=Value(dtype="int8"), length=-1),
            "labels": Sequence(feature=Value(dtype="int64"), length=-1),
        }
    )

    hf_dataset = Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        },
        features=features,
    )

    log_info(
        f"Saving processed dataset with {len(hf_dataset)} examples to {script_args.output_path}..."
    )
    hf_dataset.save_to_disk(script_args.output_path)

    # --- Post-process dataset_info.json (from generate_granite_fc_examples.py) ---
    dataset_info_path = os.path.join(script_args.output_path, "dataset_info.json")
    if os.path.exists(dataset_info_path):
        log_info(
            f"Checking and fixing {dataset_info_path} for missing 'length' in Sequence features..."
        )
        try:
            with open(dataset_info_path, "r") as f:
                dataset_info_content = json.load(f)
            modified = False
            if "features" in dataset_info_content:
                for feature_name, feature_def_dict in dataset_info_content[
                    "features"
                ].items():
                    if (
                        isinstance(feature_def_dict, dict)
                        and feature_def_dict.get("_type") == "Sequence"
                    ):
                        if "length" not in feature_def_dict:
                            log_info(
                                f"Adding missing 'length': -1 to Sequence feature '{feature_name}' in {dataset_info_path}"
                            )
                            feature_def_dict["length"] = -1
                            modified = True
            if modified:
                with open(dataset_info_path, "w") as f:
                    json.dump(dataset_info_content, f, indent=2)
                log_info(
                    f"Successfully updated {dataset_info_path} with missing 'length' fields."
                )
            else:
                log_info(
                    f"{dataset_info_path} already compliant or no Sequence features found needing 'length'."
                )
        except Exception as e:
            log_error(f"Error processing {dataset_info_path}: {e}")
    else:
        log_error(
            f"Error: {dataset_info_path} not found after saving dataset. Cannot apply fix for 'length' field."
        )

    # --- Attempt to fix Arrow metadata (from generate_granite_fc_examples.py) ---
    log_info(
        f"Attempting to fix Arrow metadata in {script_args.output_path} by reloading and re-saving..."
    )
    try:
        dataset_loaded_after_patch = Dataset.load_from_disk(script_args.output_path)
        log_info(
            f"Successfully reloaded dataset from {script_args.output_path} after patching dataset_info.json."
        )
        log_info(
            f"Features from reloaded dataset (before cast): {dataset_loaded_after_patch.features}"
        )

        log_info(f"Casting reloaded dataset to known correct features: {features}")
        dataset_for_resave = dataset_loaded_after_patch.cast(features)
        log_info(
            f"Features from dataset after cast (pre-info update): {dataset_for_resave.features}"
        )

        if dataset_for_resave.info is not None:
            dataset_for_resave.info.features = features
            log_info(
                "Updated dataset_for_resave.info.features with known correct features."
            )
        else:
            log_info(
                "dataset_for_resave.info was None, creating new DatasetInfo object."
            )
            dataset_for_resave.info = DatasetInfo(features=features)

        log_info(
            f"Features from dataset after cast and info update (to be re-saved): {dataset_for_resave.features}"
        )
        if dataset_for_resave.info:
            log_info(
                f"DatasetInfo.features after cast and info update: {dataset_for_resave.info.features}"
            )

        temp_resave_path = script_args.output_path + "__resaved_temp"
        if os.path.exists(temp_resave_path):
            shutil.rmtree(temp_resave_path)
        os.makedirs(temp_resave_path, exist_ok=True)

        log_info(f"Re-saving dataset to temporary path: {temp_resave_path}")
        dataset_for_resave.save_to_disk(temp_resave_path)
        log_info(
            f"Dataset re-saved to {temp_resave_path}. Now replacing original files."
        )

        for item_name in os.listdir(temp_resave_path):
            source_item_path = os.path.join(temp_resave_path, item_name)
            destination_item_path = os.path.join(script_args.output_path, item_name)
            if os.path.isfile(source_item_path):
                if os.path.exists(destination_item_path) and os.path.isdir(
                    destination_item_path
                ):
                    shutil.rmtree(destination_item_path)
                elif os.path.exists(destination_item_path) and os.path.isfile(
                    destination_item_path
                ):
                    os.remove(destination_item_path)
                shutil.copy2(source_item_path, destination_item_path)

        shutil.rmtree(temp_resave_path)
        log_info(
            f"Original dataset at {script_args.output_path} updated with re-saved version. Arrow metadata should now be fixed."
        )
    except Exception as e:
        log_error(f"Error during Arrow metadata fix attempt (reload and re-save): {e}")
        log_error(
            f"The dataset at {script_args.output_path} might still have problematic Arrow metadata."
        )

    log_info(
        f"--- Dataset processing for {script_args.hf_dataset_name} complete. Output at {script_args.output_path} ---"
    )


if __name__ == "__main__":
    main()
