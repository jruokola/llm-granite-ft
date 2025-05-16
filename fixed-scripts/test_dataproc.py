import argparse
import json
import os
import re  # For parsing the new dataset format
import sys
import time

from datasets import load_dataset  # Re-add this import
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

# Markers from the glaiveai/glaive-function-calling-v2 dataset
SYSTEM_MARKER_GLAIVE = "SYSTEM:"
USER_MARKER_GLAIVE = "USER:"
ASSISTANT_MARKER_GLAIVE = "ASSISTANT:"
TOOL_CALL_MARKER_GLAIVE = "<functioncall>"
FUNCTION_RESPONSE_MARKER_GLAIVE = "FUNCTION RESPONSE:"
END_OF_TEXT_GLAIVE = "<|endoftext|>"


def format_granite_turn(role, content):
    return f"{SOT}{role}{EOTR}{content}{EOTXT}\n"


def _create_labels_granite(input_ids_list, tokenizer):
    labels = [-100] * len(input_ids_list)
    decoded_full_text = tokenizer.decode(input_ids_list, skip_special_tokens=False)

    # Create a mapping from character index to token index
    token_spans = []  # list of (token_text, start_char_idx, end_char_idx)
    current_char_pos = 0
    for token_id in input_ids_list:
        # Handle cases where a single ID might decode to empty or special string
        # For robust length, decode then take len. Some special tokens might decode to empty.
        decoded_token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        token_len = len(decoded_token_text)
        token_spans.append(
            (decoded_token_text, current_char_pos, current_char_pos + token_len)
        )
        current_char_pos += token_len

    # Ensure last token span reaches end of decoded_full_text if lengths mismatch due to decode nuances
    if token_spans and token_spans[-1][2] < len(decoded_full_text):
        # This can happen if trailing special tokens are handled oddly by decode on single tokens
        # For simplicity, we'll assume the spans cover the decoded_full_text.
        # A mismatch here indicates tokenizer complexity not fully handled by this span generation.
        pass

    assistant_turn_start_str = f"{SOT}{ROLE_ASSISTANT_GRANITE}{EOTR}"

    current_search_char_idx = 0
    while current_search_char_idx < len(decoded_full_text):
        next_assistant_turn_char_idx = decoded_full_text.find(
            assistant_turn_start_str, current_search_char_idx
        )
        if next_assistant_turn_char_idx == -1:
            break

        # Content starts after the assistant role marker
        content_start_char_idx = next_assistant_turn_char_idx + len(
            assistant_turn_start_str
        )

        next_eotxt_char_idx = decoded_full_text.find(EOTXT, content_start_char_idx)
        if next_eotxt_char_idx == -1:  # Should ideally not happen
            next_eotxt_char_idx = len(decoded_full_text)

        # The content to unmask is between content_start_char_idx and next_eotxt_char_idx
        unmask_char_start = content_start_char_idx
        unmask_char_end = (
            next_eotxt_char_idx  # Unmask content up to (but not including) EOTXT
        )

        # Convert char_start and char_end to token indices
        start_token_idx = -1
        end_token_idx = -1

        for i, span_info in enumerate(token_spans):
            _, token_char_start, token_char_end = span_info
            # Check if this token is the first token at or after unmask_char_start
            if (
                start_token_idx == -1 and token_char_end > unmask_char_start
            ):  # token_char_start >= unmask_char_start also works
                start_token_idx = i

            # Check if this token is the last token at or before unmask_char_end
            if start_token_idx != -1 and token_char_start < unmask_char_end:
                end_token_idx = i  # This token is still part of the content to unmask

            if (
                token_char_start >= unmask_char_end and start_token_idx != -1
            ):  # Passed the unmask region
                break

        if start_token_idx != -1 and end_token_idx != -1:
            for i in range(start_token_idx, end_token_idx + 1):
                if i < len(labels):
                    labels[i] = input_ids_list[i]

        current_search_char_idx = next_eotxt_char_idx + len(EOTXT)

    return labels


def preprocess_example(example, tokenizer, max_length):
    raw_text = example.get("text", "")
    if not raw_text:
        return preprocess_example(
            {"text": "USER: Hello\nASSISTANT: Hi there!<|endoftext|>"},
            tokenizer,
            max_length,
        )

    formatted_text_parts = []

    system_content_full = ""
    tools_json_str = "[]"

    system_match = re.search(
        rf"^{SYSTEM_MARKER_GLAIVE}(.*?)(?=\n{USER_MARKER_GLAIVE}|$)",
        raw_text,
        re.DOTALL | re.MULTILINE,
    )
    if system_match:
        system_content_full = system_match.group(1).strip()

        json_tool_match = re.findall(
            r"(\{.*?\})|(\[.*?\])", system_content_full, re.DOTALL
        )

        actual_system_prompt_text = system_content_full
        if json_tool_match:
            for group1, group2 in reversed(json_tool_match):
                potential_json_str = group1 if group1 else group2
                try:
                    parsed_json = json.loads(potential_json_str)
                    if isinstance(parsed_json, list) and all(
                        isinstance(item, dict) and "name" in item
                        for item in parsed_json
                    ):
                        tools_json_str = potential_json_str
                        actual_system_prompt_text = actual_system_prompt_text.replace(
                            tools_json_str, ""
                        ).strip()
                        break
                    elif isinstance(parsed_json, dict) and "name" in parsed_json:
                        tools_json_str = f"[{potential_json_str}]"
                        actual_system_prompt_text = actual_system_prompt_text.replace(
                            potential_json_str, ""
                        ).strip()
                        break
                except json.JSONDecodeError:
                    continue

        if actual_system_prompt_text:
            formatted_text_parts.append(
                format_granite_turn(ROLE_SYSTEM_GRANITE, actual_system_prompt_text)
            )
        formatted_text_parts.append(
            format_granite_turn(ROLE_AVAILABLE_TOOLS_GRANITE, tools_json_str)
        )

        raw_text = raw_text[system_match.end() :].strip()
    else:
        formatted_text_parts.append(
            format_granite_turn(ROLE_AVAILABLE_TOOLS_GRANITE, "[]")
        )

    turn_pattern = re.compile(
        rf"({USER_MARKER_GLAIVE}|{ASSISTANT_MARKER_GLAIVE}|{FUNCTION_RESPONSE_MARKER_GLAIVE})\s*(.*?)\s*({TOOL_CALL_MARKER_GLAIVE}\s*(.*?)\s*)?({END_OF_TEXT_GLAIVE})?",
        re.DOTALL,
    )

    last_idx = 0
    for match in turn_pattern.finditer(raw_text):
        role_marker = match.group(1)
        content_before_tool_call = (match.group(2) or "").strip()
        tool_call_json_str = match.group(4)

        if match.start() > last_idx:
            inter_text = raw_text[last_idx : match.start()].strip()
            if inter_text:
                formatted_text_parts.append(
                    format_granite_turn(ROLE_USER_GRANITE, inter_text)
                )

        if role_marker == USER_MARKER_GLAIVE:
            formatted_text_parts.append(
                format_granite_turn(ROLE_USER_GRANITE, content_before_tool_call)
            )

        elif role_marker == ASSISTANT_MARKER_GLAIVE:
            assistant_content = ""
            if tool_call_json_str:
                try:
                    parsed_glaive_tool_call = json.loads(tool_call_json_str)
                    if "arguments" in parsed_glaive_tool_call and isinstance(
                        parsed_glaive_tool_call["arguments"], dict
                    ):
                        parsed_glaive_tool_call["arguments"] = json.dumps(
                            parsed_glaive_tool_call["arguments"]
                        )
                    granite_tool_call_str = f"{TOOL_CALL_MARKER_GRANITE}[{json.dumps(parsed_glaive_tool_call)}]"
                    assistant_content = granite_tool_call_str
                except json.JSONDecodeError:
                    eprint_rank0(
                        f"[WARNING] Failed to parse assistant tool call JSON: {tool_call_json_str}"
                    )
                    assistant_content = content_before_tool_call
            elif content_before_tool_call:
                assistant_content = content_before_tool_call
            formatted_text_parts.append(
                format_granite_turn(ROLE_ASSISTANT_GRANITE, assistant_content)
            )

        elif role_marker == FUNCTION_RESPONSE_MARKER_GLAIVE:
            formatted_text_parts.append(
                format_granite_turn(
                    ROLE_TOOL_RESPONSE_GRANITE, content_before_tool_call
                )
            )

        last_idx = match.end()

    if last_idx < len(raw_text):
        trailing_text = raw_text[last_idx:].strip()
        if trailing_text:
            formatted_text_parts.append(
                format_granite_turn(ROLE_USER_GRANITE, trailing_text)
            )

    final_formatted_text = "".join(formatted_text_parts)

    if not final_formatted_text.strip():
        print_rank0(
            f"[WARNING] Example resulted in empty formatted text after parsing. Using fallback. Original: {example['text'][:500]}"
        )
        final_formatted_text = format_granite_turn(
            ROLE_USER_GRANITE, "Hello"
        ) + format_granite_turn(
            ROLE_ASSISTANT_GRANITE, "Hello! How can I help you today?"
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

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Glaive FunctionCalling V2 dataset for Granite and save to disk."
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

    special_tokens_to_add = [SOT, EOTR, EOTXT, TOOL_CALL_MARKER_GRANITE]

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print_rank0(
                f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}"
            )
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            print_rank0("Added [PAD] as pad_token.")

    print_rank0(f"Loading raw dataset: {script_args.dataset_name} (default config)")
    try:
        raw_dataset_train = load_dataset(script_args.dataset_name, split="train")
    except Exception as e:
        eprint_rank0(f"Failed to load dataset: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    print_rank0(f"Raw 'train' dataset loaded. Total examples: {len(raw_dataset_train)}")

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
        remove_columns=raw_dataset_subset.column_names,
        desc="Preprocessing dataset for Granite format",
    )

    map_duration = time.time() - start_map_time
    print_rank0(f"Dataset preprocessing finished in {map_duration:.2f} seconds.")

    print_rank0(f"Saving processed dataset to {script_args.output_path}...")
    start_save_time = time.time()
    processed_dataset.save_to_disk(script_args.output_path)
    save_duration = time.time() - start_save_time
    print_rank0(f"Processed dataset saved in {save_duration:.2f} seconds.")
    print_rank0("--- Preprocessing Complete ---")
