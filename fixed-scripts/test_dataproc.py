import argparse
import json
import os
import sys
import time

from datasets import load_dataset
from transformers import AutoTokenizer


def print_rank0(*args, **kwargs):
    if int(os.getenv("RANK", "0")) == 0:
        print(*args, **kwargs)
        sys.stdout.flush()


def eprint_rank0(*args, **kwargs):
    if int(os.getenv("RANK", "0")) == 0:
        print(*args, file=sys.stderr, **kwargs)
        sys.stderr.flush()


# --- Granite Specific Tokens ---
SOT = "<|start_of_role|>"
EOTR = "<|end_of_role|>"
EOTXT = "<|end_of_text|>"
TOOL_CALL_MARKER_GRANITE = "<|tool_call|>"

# Roles for Granite
ROLE_SYSTEM_GRANITE = "system"
ROLE_AVAILABLE_TOOLS_GRANITE = "available_tools"
ROLE_USER_GRANITE = "user"
ROLE_ASSISTANT_GRANITE = "assistant"
ROLE_TOOL_RESPONSE_GRANITE = "tool_response"


def format_granite_turn(role, content):
    content_str = str(content).strip()
    return f"{SOT}{role}{EOTR}{content_str}{EOTXT}\n"


def _create_labels_granite(input_ids_list, tokenizer):
    labels = [-100] * len(input_ids_list)

    # Attempt to find assistant turns and unmask their content.
    # This relies on the Granite-specific formatting being present in the input_ids.

    # Convert input_ids to a list of tokens for easier debugging and subsequence finding
    # This is for conceptual matching; actual matching should use IDs if special tokens are single IDs.
    # However, role names like "assistant" are multi-token.

    # We will iterate through the tokens and identify segments belonging to the assistant.
    # A segment starts after SOT + assistant + EOTR and ends before the turn's EOTXT.

    # Create a string representation to find character indices first (heuristic)
    decoded_full_text = tokenizer.decode(
        input_ids_list, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    # Build token_spans: list of (start_char_offset, end_char_offset) for each token_id
    # This helps map character-based findings back to token indices.
    token_offsets = []
    current_offset = 0
    for token_id in input_ids_list:
        # It's important that individual token decoding matches how full sequence decoding works.
        # Using clean_up_tokenization_spaces=False above and here might help consistency.
        decoded_token = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        token_length = len(decoded_token)
        token_offsets.append((current_offset, current_offset + token_length))
        current_offset += token_length

    # Ensure consistency if decode adds/removes prefix/suffix spaces for the whole sequence
    # This is a common pain point. If `current_offset != len(decoded_full_text)`, mapping is tricky.

    assistant_turn_start_marker = f"{SOT}{ROLE_ASSISTANT_GRANITE}{EOTR}"

    search_start_char = 0
    while search_start_char < len(decoded_full_text):
        assistant_marker_char_start = decoded_full_text.find(
            assistant_turn_start_marker, search_start_char
        )
        if assistant_marker_char_start == -1:
            break  # No more assistant turns

        # The actual content to unmask starts *after* this marker
        unmask_content_char_start = assistant_marker_char_start + len(
            assistant_turn_start_marker
        )

        # Find the end of this assistant's turn (the EOTXT for this turn)
        unmask_content_char_end = decoded_full_text.find(
            EOTXT, unmask_content_char_start
        )
        if unmask_content_char_end == -1:  # Should not happen for well-formed turns
            unmask_content_char_end = len(decoded_full_text)
            # This might unmask till the end if EOTXT is missing, which could be padding.

        # Convert character start/end to token start/end indices
        start_token_idx = -1
        end_token_idx = -1

        for i, (tok_start_char, tok_end_char) in enumerate(token_offsets):
            # First token whose end is after our content start
            if start_token_idx == -1 and tok_end_char > unmask_content_char_start:
                start_token_idx = i

            # Last token whose start is before our content end
            if start_token_idx != -1 and tok_start_char < unmask_content_char_end:
                end_token_idx = i

            if tok_start_char >= unmask_content_char_end and start_token_idx != -1:
                break  # We've passed the content region

        if (
            start_token_idx != -1
            and end_token_idx != -1
            and end_token_idx >= start_token_idx
        ):
            # Check if the content is just padding (e.g., all EOS/PAD tokens)
            # This is a heuristic; pad_token_id might not be eos_token_id
            is_padding_content = True
            for i in range(start_token_idx, end_token_idx + 1):
                if (
                    input_ids_list[i] != tokenizer.pad_token_id
                    and input_ids_list[i] != tokenizer.eos_token_id
                ):  # Check against both
                    is_padding_content = False
                    break

            if not is_padding_content:
                for i in range(start_token_idx, end_token_idx + 1):
                    if i < len(labels):  # Ensure within bounds
                        labels[i] = input_ids_list[i]

        search_start_char = unmask_content_char_end + len(
            EOTXT
        )  # Move search past this EOTXT

    return labels


def preprocess_example(example, tokenizer, max_length):
    example_id_for_log = example.get(
        "idx_col", example.get("id", "UNKNOWN_ID_IN_PREPROC")
    )
    raw_text_initial = example.get(
        "text", ""
    )  # For hqfx, this field is not used directly.

    # --- BEGIN DEBUG PRINT ---
    if (
        int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5
    ):  # Limit debug prints
        print_rank0(
            f"\n--- PREPROCESSING Example ID: {example_id_for_log} (hqfx format) ---"
        )
        print_rank0(f"SYSTEM FIELD: '''{str(example.get('system', ''))[:200]}...'''")
        print_rank0(f"TOOLS FIELD: '''{str(example.get('tools', ''))[:200]}...'''")
        chat_sample = example.get("chat", [])
        if isinstance(chat_sample, str):  # If chat is a string, print part of it
            print_rank0(f"CHAT FIELD (string): '''{chat_sample[:200]}...'''")
        elif isinstance(chat_sample, list) and chat_sample:  # If list, print first turn
            print_rank0(
                f"CHAT FIELD (first turn if any): '''{str(chat_sample[0])[:200]}...'''"
            )
        else:
            print_rank0(f"CHAT FIELD: {type(chat_sample)}")
    # --- END DEBUG PRINT ---

    formatted_text_parts = []

    system_content = example.get("system")
    if system_content and str(system_content).strip():
        formatted_text_parts.append(
            format_granite_turn(ROLE_SYSTEM_GRANITE, str(system_content))
        )

    tools_data = example.get("tools")
    tools_json_str = "[]"
    if tools_data:
        if isinstance(tools_data, str):
            try:
                parsed_tools = json.loads(tools_data)
                tools_json_str = json.dumps(parsed_tools)
            except json.JSONDecodeError:
                if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
                    eprint_rank0(
                        f"[WARNING] Example {example_id_for_log}: Tools field is a string but not valid JSON: {tools_data[:200]}..."
                    )
                tools_json_str = "[]"
        elif isinstance(tools_data, list):
            tools_json_str = json.dumps(tools_data)
        else:  # Should not happen with hqfx
            if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
                eprint_rank0(
                    f"[WARNING] Example {example_id_for_log}: Tools field has unexpected type: {type(tools_data)}. Content: {str(tools_data)[:200]}..."
                )

    formatted_text_parts.append(
        format_granite_turn(ROLE_AVAILABLE_TOOLS_GRANITE, tools_json_str)
    )

    chat_data_raw = example.get("chat")
    chat_history = []
    if isinstance(chat_data_raw, str):
        try:
            chat_history = json.loads(chat_data_raw)
            if not isinstance(chat_history, list):
                if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
                    eprint_rank0(
                        f"[WARNING] Example {example_id_for_log}: 'chat' field (string) did not parse to a list: {chat_data_raw[:200]}..."
                    )
                chat_history = []
        except json.JSONDecodeError:
            if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
                eprint_rank0(
                    f"[WARNING] Example {example_id_for_log}: 'chat' field is a string but not valid JSON: {chat_data_raw[:200]}..."
                )
            chat_history = []
    elif isinstance(chat_data_raw, list):
        chat_history = chat_data_raw

    if not chat_history:
        if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
            print_rank0(
                f"[INFO] Example ID {example_id_for_log} has no valid chat history. Original 'chat' field type: {type(chat_data_raw)}."
            )
        # If only system/tools were present, formatted_text_parts might not be empty.
        # If all key fields (system, tools, chat) are effectively empty, then return None.
        if (
            not system_content and not tools_data
        ):  # Check if original system/tools were also empty
            if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
                print_rank0(
                    f"[WARNING] Example ID {example_id_for_log}: All key fields (system, tools, chat) are effectively empty. Returning None."
                )
            return None

    for turn in chat_history:
        role = turn.get("role", "").lower()
        content = turn.get("content")
        function_call = turn.get("function_call")

        if role == "user":
            formatted_text_parts.append(
                format_granite_turn(
                    ROLE_USER_GRANITE, str(content).strip() if content else ""
                )
            )
        elif role == "assistant":
            assistant_response = ""
            if function_call and isinstance(function_call, dict):
                if "arguments" in function_call and isinstance(
                    function_call["arguments"], dict
                ):  # hqfx arguments are already strings
                    function_call["arguments"] = json.dumps(function_call["arguments"])
                assistant_response += (
                    f"{TOOL_CALL_MARKER_GRANITE}[{json.dumps(function_call)}]"
                )
            elif content:
                assistant_response += str(content).strip()
            formatted_text_parts.append(
                format_granite_turn(ROLE_ASSISTANT_GRANITE, assistant_response)
            )
        elif role == "tool_output" or role == "tool_response":
            tool_output_content = str(content).strip() if content else "{}"
            try:
                json.loads(tool_output_content)
            except json.JSONDecodeError:
                if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
                    eprint_rank0(
                        f"[WARNING] Example {example_id_for_log}: tool_output content is not valid JSON: {tool_output_content[:200]}..."
                    )
            formatted_text_parts.append(
                format_granite_turn(ROLE_TOOL_RESPONSE_GRANITE, tool_output_content)
            )

    final_formatted_text = "".join(formatted_text_parts)

    if not final_formatted_text.strip():  # Should be rare if above logic is fine
        if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
            print_rank0(
                f"[WARNING] Example ID {example_id_for_log} resulted in empty final_formatted_text after parsing. Original example: {str(example)[:500]}. Returning None."
            )
        return None

    if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
        print_rank0(
            f"\nFINAL FORMATTED TEXT for Example ID {example_id_for_log} (before tokenization):\n'''{final_formatted_text[:1000]}...'''\n"
        )

    encodings = tokenizer(
        final_formatted_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = _create_labels_granite(input_ids, tokenizer)

    output_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
        print_rank0(
            f"Output dict keys for Example ID {example_id_for_log}: {list(output_dict.keys())}"
        )
        if not input_ids or len(input_ids) < 5:
            print_rank0(
                f"[WARNING] Example ID {example_id_for_log} has very short or empty input_ids: {input_ids}"
            )

    return output_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess hqfx/glaive_fc_v2 dataset for Granite and save to disk."
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
        default="hqfx/glaive_fc_v2",
        help="Name of the dataset on Hugging Face Hub.",
    )
    # ... (rest of __main__ is unchanged) ...
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
        default=None,
        help="Number of processes to use for .map(). Defaults to available CPUs.",
    )
    script_args = parser.parse_args()

    if script_args.num_proc is None:
        script_args.num_proc = os.cpu_count()
    print_rank0(f"Using {script_args.num_proc} processes for dataset mapping.")

    print_rank0(f"Loading tokenizer: {script_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print_rank0(
                f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}"
            )
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            print_rank0("Added [PAD] as pad_token.")

    print_rank0(f"Loading raw dataset: {script_args.dataset_name}")
    try:
        raw_dataset_train = load_dataset(script_args.dataset_name, split="train")
    except Exception as e:
        eprint_rank0(f"Failed to load dataset: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    print_rank0(f"Raw 'train' dataset loaded. Total examples: {len(raw_dataset_train)}")

    raw_dataset_train = raw_dataset_train.add_column(
        "idx_col", range(len(raw_dataset_train))
    )

    if (
        script_args.num_samples_to_process > 0
        and script_args.num_samples_to_process < len(raw_dataset_train)
    ):
        raw_dataset_subset = raw_dataset_train.select(
            range(script_args.num_samples_to_process)
        )
        print_rank0(f"Processing a subset of {len(raw_dataset_subset)} samples.")
    else:
        raw_dataset_subset = raw_dataset_train
        print_rank0(
            f"Processing all {len(raw_dataset_subset)} samples from 'train' split."
        )

    print_rank0(
        f"Starting dataset preprocessing with {script_args.num_proc} processes..."
    )
    start_map_time = time.time()

    processed_dataset = raw_dataset_subset.map(
        preprocess_example,
        fn_kwargs={"tokenizer": tokenizer, "max_length": script_args.max_seq_length},
        num_proc=script_args.num_proc,
        desc="Preprocessing dataset for Granite format (hqfx)",
    )

    original_count = len(processed_dataset)
    processed_dataset = processed_dataset.filter(
        lambda example: example is not None
        and "input_ids" in example
        and example["input_ids"] is not None
        and "attention_mask" in example
        and example["attention_mask"] is not None
        and "labels" in example
        and example["labels"] is not None
    )
    filtered_count = len(processed_dataset)
    print_rank0(
        f"Number of examples after filtering None/incomplete: {filtered_count} (removed {original_count - filtered_count})"
    )

    if filtered_count > 0:
        final_columns = ["input_ids", "attention_mask", "labels"]
        actual_present_cols = [
            col for col in final_columns if col in processed_dataset.column_names
        ]
        if len(actual_present_cols) != len(final_columns):
            eprint_rank0(
                f"[ERROR] Not all expected columns ({final_columns}) are present in the processed data. Found: {processed_dataset.column_names}. Aborting save."
            )
            sys.exit(1)

        processed_dataset = processed_dataset.select_columns(final_columns)

        print_rank0(f"Dataset features before saving: {processed_dataset.features}")
        print_rank0(
            f"First example before saving: {processed_dataset[0] if filtered_count > 0 else 'Dataset is empty after filtering'}"
        )

        print_rank0(f"Saving processed dataset to {script_args.output_path}...")
        start_save_time = time.time()
        processed_dataset.save_to_disk(script_args.output_path)
        save_duration = time.time() - start_save_time
        print_rank0(f"Processed dataset saved in {save_duration:.2f} seconds.")
    else:
        print_rank0(
            "[ERROR] No valid examples were processed and passed filtering. Dataset will not be saved."
        )

    print_rank0("--- Preprocessing Complete ---")
