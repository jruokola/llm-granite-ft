import argparse
import json
import os
import re  # For parsing the new dataset format
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
TOOL_CALL_MARKER_GRANITE = "<|tool_call|>"  # Granite's specific token

# Roles for Granite
ROLE_SYSTEM_GRANITE = "system"
ROLE_AVAILABLE_TOOLS_GRANITE = "available_tools"
ROLE_USER_GRANITE = "user"
ROLE_ASSISTANT_GRANITE = "assistant"
ROLE_TOOL_RESPONSE_GRANITE = "tool_response"

# Markers from the new glaiveai/glaive-function-calling-v2 dataset
# Note: The dataset viewer shows "SYSTEM:", "USER:", "ASSISTANT:", "FUNCTION RESPONSE:"
# and "<functioncall>" (lowercase)
SYSTEM_MARKER_GLAIVE = "SYSTEM:"
USER_MARKER_GLAIVE = "USER:"
ASSISTANT_MARKER_GLAIVE = "ASSISTANT:"
TOOL_CALL_MARKER_GLAIVE = "<functioncall>"  # Lowercase in example
FUNCTION_RESPONSE_MARKER_GLAIVE = "FUNCTION RESPONSE:"
END_OF_TEXT_GLAIVE = "<|endoftext|>"  # Used by assistant turns in glaive


def format_granite_turn(role, content):
    return f"{SOT}{role}{EOTR}{content}{EOTXT}\n"


def _create_labels_granite(input_ids_list, tokenizer):
    labels = [-100] * len(input_ids_list)
    decoded_full_text = tokenizer.decode(input_ids_list, skip_special_tokens=False)

    # Heuristic to unmask assistant responses and tool calls in Granite format
    # This needs to be robust to the tokenizer's handling of special tokens.

    assistant_turn_start_str = f"{SOT}{ROLE_ASSISTANT_GRANITE}{EOTR}"
    # TOOL_CALL_MARKER_GRANITE is "<|tool_call|>"

    current_search_idx = 0
    while current_search_idx < len(decoded_full_text):
        next_assistant_turn_char_idx = decoded_full_text.find(
            assistant_turn_start_str, current_search_idx
        )
        if next_assistant_turn_char_idx == -1:
            break  # No more assistant turns

        # Content starts after the assistant role marker
        content_start_char_idx = next_assistant_turn_char_idx + len(
            assistant_turn_start_str
        )

        # Find the end of this assistant's turn
        next_eotxt_char_idx = decoded_full_text.find(EOTXT, content_start_char_idx)
        if next_eotxt_char_idx == -1:  # Should not happen if data is well-formed
            next_eotxt_char_idx = len(decoded_full_text)

        # Determine if it's a tool call or text response
        # The TOOL_CALL_MARKER_GRANITE should appear immediately after assistant role marker if it's a tool call
        is_tool_call = decoded_full_text[content_start_char_idx:].startswith(
            TOOL_CALL_MARKER_GRANITE
        )

        unmask_char_start = -1
        unmask_char_end = next_eotxt_char_idx  # Unmask up to EOTXT

        if is_tool_call:
            unmask_char_start = (
                content_start_char_idx  # Start unmasking from <|tool_call|>
            )
        else:  # Textual response
            unmask_char_start = content_start_char_idx  # Start unmasking from content

        # Convert character indices to token indices (approximate and potentially slow)
        # This is a common challenge. A more robust way is to tokenize segments.
        start_token_idx, end_token_idx = -1, -1
        current_char_offset = 0

        # Find start_token_idx
        for i, token_id in enumerate(input_ids_list):
            # We need to handle cases where special tokens are split by the tokenizer
            # For this heuristic, we decode single tokens.
            try:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            except:  # Broad exception for safety, tokenizers can be tricky
                token_text = " "  # Placeholder if decode fails

            if current_char_offset >= unmask_char_start and start_token_idx == -1:
                start_token_idx = i

            current_char_offset += len(token_text)

            if current_char_offset >= unmask_char_end and start_token_idx != -1:
                end_token_idx = i
                break

        if start_token_idx != -1:
            # If end_token_idx wasn't found (e.g. unmask_char_end was end of string)
            if end_token_idx == -1 and unmask_char_end == len(decoded_full_text):
                end_token_idx = len(input_ids_list) - 1

            if end_token_idx != -1:
                for i in range(start_token_idx, end_token_idx + 1):  # Inclusive
                    if i < len(labels):  # Boundary check
                        labels[i] = input_ids_list[i]

        current_search_idx = next_eotxt_char_idx + len(EOTXT)

    return labels


def preprocess_example(example, tokenizer, max_length):
    # The input 'example' is a dictionary, and the relevant data is in example['text']
    raw_text = example.get("text", "")
    if not raw_text:
        # Fallback for empty raw text
        return preprocess_example(
            {"text": "USER: Hello\nASSISTANT: Hi there!<|endoftext|>"},
            tokenizer,
            max_length,
        )

    formatted_text_parts = []

    # --- 1. Extract System Prompt and Tools ---
    system_content_full = ""
    tools_json_str = "[]"  # Default to empty list of tools

    system_match = re.search(
        rf"^{SYSTEM_MARKER_GLAIVE}(.*?)(?=\n{USER_MARKER_GLAIVE}|$)",
        raw_text,
        re.DOTALL | re.MULTILINE,
    )
    if system_match:
        system_content_full = system_match.group(1).strip()

        # Try to extract JSON tool definition from system prompt
        # This assumes tools JSON is the last {...} or [...] block in the system prompt
        # More robust parsing might be needed if system prompt structure varies a lot
        json_tool_match = re.findall(
            r"(\{.*?\})|(\[.*?\])", system_content_full, re.DOTALL
        )

        actual_system_prompt_text = system_content_full
        if json_tool_match:
            # Iterate backwards to find the last valid JSON block that could be tools
            for group1, group2 in reversed(json_tool_match):
                potential_json_str = group1 if group1 else group2
                try:
                    # Check if it's a list of tool objects
                    parsed_json = json.loads(potential_json_str)
                    if isinstance(parsed_json, list) and all(
                        isinstance(item, dict) and "name" in item
                        for item in parsed_json
                    ):
                        tools_json_str = potential_json_str
                        # Remove the tool string from the system prompt text
                        actual_system_prompt_text = actual_system_prompt_text.replace(
                            tools_json_str, ""
                        ).strip()
                        break  # Found valid tools
                    elif (
                        isinstance(parsed_json, dict) and "name" in parsed_json
                    ):  # Single tool case
                        tools_json_str = f"[{potential_json_str}]"  # Wrap in list
                        actual_system_prompt_text = actual_system_prompt_text.replace(
                            potential_json_str, ""
                        ).strip()
                        break
                except json.JSONDecodeError:
                    continue  # Not a valid JSON, try previous match

        if actual_system_prompt_text:
            formatted_text_parts.append(
                format_granite_turn(ROLE_SYSTEM_GRANITE, actual_system_prompt_text)
            )
        formatted_text_parts.append(
            format_granite_turn(ROLE_AVAILABLE_TOOLS_GRANITE, tools_json_str)
        )

        # Remove the processed system part from raw_text for further parsing
        raw_text = raw_text[system_match.end() :].strip()
    else:  # No system prompt, add default empty available_tools
        formatted_text_parts.append(
            format_granite_turn(ROLE_AVAILABLE_TOOLS_GRANITE, "[]")
        )

    # --- 2. Process Chat Turns (User, Assistant, Function Response) ---
    # Split turns by looking for USER:, ASSISTANT:, FUNCTION RESPONSE:
    # This regex tries to capture the role, its content, and any <functioncall> or <|endoftext|>
    turn_pattern = re.compile(
        rf"({USER_MARKER_GLAIVE}|{ASSISTANT_MARKER_GLAIVE}|{FUNCTION_RESPONSE_MARKER_GLAIVE})\s*(.*?)\s*({TOOL_CALL_MARKER_GLAIVE}\s*(.*?)\s*)?({END_OF_TEXT_GLAIVE})?",
        re.DOTALL,
    )

    last_idx = 0
    for match in turn_pattern.finditer(raw_text):
        role_marker = match.group(1)
        content_before_tool_call = (match.group(2) or "").strip()

        # tool_call_full_match = match.group(3) # e.g. "<functioncall> {...}"
        tool_call_json_str = match.group(4)  # e.g. "{...}"
        # end_of_text_marker_found = match.group(5) # e.g. "<|endoftext|>"

        # Add any text between turns as user text if not captured (less likely with this regex)
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
                    # The example shows arguments as a string within the JSON.
                    # The Granite format expects a list of tool calls.
                    # <|tool_call|>[{"name": "...", "arguments": "{...}"}]
                    # The glaive format is <functioncall> {"name": "...", "arguments": "'{...}'"}
                    # We need to parse the glaive JSON, then re-serialize for Granite.
                    parsed_glaive_tool_call = json.loads(tool_call_json_str)

                    # Ensure arguments is a string, as Granite example shows
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
                    assistant_content = (
                        content_before_tool_call  # Fallback to text if any
                    )
            elif content_before_tool_call:
                assistant_content = content_before_tool_call
            formatted_text_parts.append(
                format_granite_turn(ROLE_ASSISTANT_GRANITE, assistant_content)
            )

        elif role_marker == FUNCTION_RESPONSE_MARKER_GLAIVE:
            # Content is expected to be a JSON string.
            # The regex captures content_before_tool_call as the function response here.
            formatted_text_parts.append(
                format_granite_turn(
                    ROLE_TOOL_RESPONSE_GRANITE, content_before_tool_call
                )
            )

        last_idx = match.end()

    # Append any trailing text as a user turn (if any)
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
        default="glaiveai/glaive-function-calling-v2",  # Default to the new dataset
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

    # IMPORTANT: Ensure the tokenizer has the Granite special tokens added.
    # This step is crucial and should ideally be done when preparing the tokenizer for the main training.
    # If they are not part of the base tokenizer, they MUST be added.
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, trust_remote_code=True
    )

    special_tokens_to_add = [SOT, EOTR, EOTXT, TOOL_CALL_MARKER_GRANITE]
    # Check if they exist, add if not. This is a basic check.
    # A more robust check would be `tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id`
    # or checking against `tokenizer.added_tokens_decoder`

    # For this script, we assume the tokenizer used for fine-tuning will have these.
    # If running this script standalone to *prepare* data, one might add them here:
    # num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
    # if num_added_toks > 0:
    #     print_rank0(f"Added {num_added_toks} special tokens to tokenizer: {special_tokens_to_add}")
    # Note: If tokens are added, the model's embedding layer usually needs to be resized.
    # This script only processes data, so resizing is for the main training script.

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print_rank0(
                f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}"
            )
        else:  # Fallback if no EOS token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            print_rank0("Added [PAD] as pad_token.")

    print_rank0(f"Loading raw dataset: {script_args.dataset_name} (default config)")
    try:
        # For glaiveai/glaive-function-calling-v2, 'default' config and 'train' split
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
        remove_columns=raw_dataset_subset.column_names,  # Remove 'text' column
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
