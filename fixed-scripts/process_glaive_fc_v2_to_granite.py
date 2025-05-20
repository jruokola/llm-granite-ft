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
        default=512,  # Matching generate_granite_fc_examples.py for format consistency check
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        default="glaiveai/glaive-function-calling-v2",
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

        # 1. System Turn (predefined, same for all examples from this script)
        example_parts.append(format_granite_turn(ROLE_SYSTEM_GRANITE, system_prompt))

        # 2. Available Tools Turn
        # In glaiveai/glaive-function-calling-v2, tools are in the 'system' field.
        # Example: "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required - {json_func_def_1} {json_func_def_2}"
        # We need to extract these JSON function definitions.
        system_field_content = raw_example.get("system", "")
        tools_marker = "Use them if required - "
        tools_json_str = "[]"  # Default to empty list of tools
        if tools_marker in system_field_content:
            potential_tools_part = system_field_content.split(tools_marker, 1)[
                -1
            ].strip()
            # The functions are concatenated JSON objects, not a single JSON list.
            # We need to parse them individually and then wrap them in a list.
            extracted_tools = []
            decoder = json.JSONDecoder()
            idx = 0
            while idx < len(potential_tools_part):
                potential_tools_part = potential_tools_part[idx:].strip()
                if not potential_tools_part:
                    break
                try:
                    func, end_idx = decoder.raw_decode(potential_tools_part)
                    extracted_tools.append(func)
                    idx = end_idx
                except json.JSONDecodeError:
                    # This might happen if there's non-JSON text after the functions
                    # or if the concatenation isn't perfect.
                    log_warning(
                        f"Could not parse all function JSONs in example {i} from: {potential_tools_part[idx : idx + 50]}..."
                    )
                    break
            if extracted_tools:
                tools_json_str = json.dumps(extracted_tools)

        example_parts.append(
            format_granite_turn(ROLE_AVAILABLE_TOOLS_GRANITE, tools_json_str)
        )

        # 3. Conversation Turns from 'chat' field
        # Example 'chat': "USER: ... ASSISTANT: ... <|endoftext|> FUNCTION RESPONSE: ... ASSISTANT: ... <|endoftext|>"
        chat_content = raw_example.get("chat", "")
        turns_raw = chat_content.split("<|endoftext|>")

        for turn_raw in turns_raw:
            turn_raw = turn_raw.strip()
            if not turn_raw:
                continue

            if turn_raw.startswith("USER:"):
                content = turn_raw[len("USER:") :].strip()
                example_parts.append(format_granite_turn(ROLE_USER_GRANITE, content))
            elif turn_raw.startswith("ASSISTANT:"):
                content_part = turn_raw[len("ASSISTANT:") :].strip()
                if content_part.startswith("<functioncall>"):
                    # Extract JSON from <functioncall> {"name": "...", "arguments": '...'}
                    # The arguments themselves are a string that needs to be parsed if we were to use them,
                    # but for Granite format, we just need the whole function call object string.
                    # The official dataset has the arguments as a string, not a nested JSON object directly.
                    # e.g. <functioncall> {"name": "get_news_headlines", "arguments": '{"country": "United States"}'}
                    # The Granite format expects the <|tool_call|> marker followed by a JSON *list* of tool calls.
                    # The glaive format provides a single tool call object as a string.

                    fc_marker_start = "<functioncall>"
                    # The end of the function call is implicitly before <|endoftext|> or end of string
                    # For simplicity, assume the function call is the entirety of the content_part after the marker

                    fc_json_str = content_part[len(fc_marker_start) :].strip()

                    # The fc_json_str is like: {"name": "N", "arguments": '{"K":"V"}'}
                    # Granite expects: [{"name": "N", "arguments": "{\"K\":\"V\""}] (a list of such objects, stringified)
                    # So, we need to wrap fc_json_str in list brackets to make it a JSON array string.
                    # However, the arguments within the glaive format are already a string.
                    # Let's parse it to ensure it's valid JSON, then re-stringify it as a list item.
                    try:
                        # Attempt to parse the function call string to validate it's a single JSON object
                        # The arguments field is a string, which is fine for the Granite format if the outer structure is a list of calls.
                        parsed_fc = json.loads(fc_json_str)  # This should be a dict
                        # Wrap it as a list containing one item, then dump to string
                        granite_fc_list_str = json.dumps([parsed_fc])
                        assistant_response_content = (
                            f"{TOOL_CALL_MARKER_GRANITE}{granite_fc_list_str}"
                        )
                        example_parts.append(
                            format_granite_turn(
                                ROLE_ASSISTANT_GRANITE, assistant_response_content
                            )
                        )
                    except json.JSONDecodeError:
                        log_error(
                            f"Failed to parse function call JSON in example {i}: {fc_json_str}. Using raw content part."
                        )
                        example_parts.append(
                            format_granite_turn(ROLE_ASSISTANT_GRANITE, content_part)
                        )  # Fallback
                else:
                    example_parts.append(
                        format_granite_turn(ROLE_ASSISTANT_GRANITE, content_part)
                    )
            elif turn_raw.startswith("FUNCTION RESPONSE:"):
                content = turn_raw[len("FUNCTION RESPONSE:") :].strip()
                # Content is already the JSON string of the tool output.
                example_parts.append(
                    format_granite_turn(ROLE_TOOL_RESPONSE_GRANITE, content)
                )
            else:
                log_warning(
                    f"Unrecognized turn structure in example {i}: {turn_raw[:100]}..."
                )

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

    log_info("Creating intermediate dataset from formatted text examples...")
    # Create a temporary dataset with just the formatted text
    temp_data = {"text": formatted_text_examples}
    temp_dataset = Dataset.from_dict(temp_data)

    log_info("Tokenizing and creating labels in parallel using Dataset.map()...")

    def tokenize_and_label_batch(examples):
        # Tokenize the batch of text examples
        batch_encodings = tokenizer(
            examples["text"],
            max_length=script_args.max_seq_length,
            truncation=True,
            padding="max_length",  # Padding will be applied per batch
            return_attention_mask=True,
        )

        batch_labels = []
        for i in range(len(batch_encodings["input_ids"])):
            input_ids = batch_encodings["input_ids"][i]
            labels = create_labels_for_granite_sequence(input_ids, tokenizer)
            batch_labels.append(labels)

        batch_encodings["labels"] = batch_labels
        return batch_encodings

    # Determine num_proc. os.cpu_count() can be a good default.
    # Let's use a reasonable default, e.g., 4, or allow it to be configured.
    # For now, let's pick a sensible default or use a fraction of available CPUs.
    num_cpus = os.cpu_count()
    num_proc_to_use = max(1, num_cpus // 2 if num_cpus else 1)  # Use half CPUs, min 1
    log_info(f"Using {num_proc_to_use} processes for mapping.")

    # Define features for the final dataset
    final_features = Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int32"), length=-1),
            "attention_mask": Sequence(feature=Value(dtype="int8"), length=-1),
            "labels": Sequence(feature=Value(dtype="int64"), length=-1),
        }
    )

    # Apply the mapping function
    # The `map` function will return a new Dataset with the added columns.
    # We need to ensure the output features are correctly specified if `map` doesn't infer them perfectly,
    # or cast them afterwards. However, returning a dict usually works well.
    processed_dataset = temp_dataset.map(
        tokenize_and_label_batch,
        batched=True,
        num_proc=num_proc_to_use,
        remove_columns=["text"],  # Remove the original text column
        features=final_features,  # Explicitly set features for the output dataset
        desc="Tokenizing and labeling",
    )

    # Ensure the features are set correctly on the final dataset object
    # This might be redundant if `features` arg in `map` works as expected, but good for safety.
    if processed_dataset.features is None or set(
        processed_dataset.features.keys()
    ) != set(final_features.keys()):
        log_info(f"Casting dataset to final features: {final_features}")
        hf_dataset = processed_dataset.cast(final_features)
    else:
        hf_dataset = processed_dataset

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

        log_info(
            f"Casting reloaded dataset to known correct features: {final_features}"
        )  # Use final_features
        dataset_for_resave = dataset_loaded_after_patch.cast(
            final_features
        )  # Use final_features
        log_info(
            f"Features from dataset after cast (pre-info update): {dataset_for_resave.features}"
        )

        if dataset_for_resave.info is not None:
            dataset_for_resave.info.features = final_features  # Use final_features
            log_info(
                "Updated dataset_for_resave.info.features with known correct features."
            )
        else:
            log_info(
                "dataset_for_resave.info was None, creating new DatasetInfo object."
            )
            dataset_for_resave.info = DatasetInfo(
                features=final_features
            )  # Use final_features

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
