import argparse
import ast  # For literal_eval
import json
import os
import re
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
ROLE_TOOL_RESPONSE_GRANITE = "tool_response"  # For Granite output

# Roles from Hermes dataset
ROLE_SYSTEM_HERMES = "system"
ROLE_HUMAN_HERMES = "human"
ROLE_GPT_HERMES = "gpt"


def format_granite_turn(role, content):
    content_str = str(content).strip()
    return f"{SOT}{role}{EOTR}{content_str}{EOTXT}\n"


def _create_labels_granite(input_ids_list, tokenizer):
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
            is_purely_padding_content = True
            if start_token_idx <= end_token_idx:  # Ensure there's a segment
                for i_check in range(start_token_idx, end_token_idx + 1):
                    if input_ids_list[i_check] != tokenizer.pad_token_id:
                        is_purely_padding_content = False
                        break
            else:  # No valid segment found, treat as padding or empty
                is_purely_padding_content = True

            if not is_purely_padding_content:
                for i_label in range(start_token_idx, end_token_idx + 1):
                    if i_label < len(labels):  # boundary check
                        labels[i_label] = input_ids_list[i_label]
        search_start_char = unmask_content_char_end + len(EOTXT)
    return labels


def preprocess_example(example, tokenizer, max_length):
    example_id_for_log = example.get(
        "idx_col", example.get("id", "UNKNOWN_ID_IN_PREPROC")
    )
    conversation_turns = example.get("conversations")

    if not conversation_turns or not isinstance(conversation_turns, list):
        if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
            print_rank0(
                f"[WARNING] Example ID {example_id_for_log}: 'conversations' field is missing, not a list, or empty. Skipping."
            )
        return None

    formatted_text_parts = []

    if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
        print_rank0(
            f"\n--- PREPROCESSING Example ID: {example_id_for_log} (NousResearch format) ---"
        )
        if conversation_turns:
            print_rank0(f"First turn raw: {str(conversation_turns[0])[:500]}...")

    for turn_idx, turn_data in enumerate(conversation_turns):
        role = turn_data.get("from", "").lower()
        value = turn_data.get("value", "")

        if role == ROLE_SYSTEM_HERMES:
            system_content_full_str = str(value).strip()
            tools_json_for_granite = "[]"  # Default for tools
            system_text_for_granite = system_content_full_str  # Default

            tools_match = re.search(
                r"<tools>\s*(.*?)\s*</tools>",  # Allow whitespace around content
                system_content_full_str,
                re.DOTALL | re.IGNORECASE,
            )
            if tools_match:
                tools_str_from_hermes = tools_match.group(1).strip()
                # Construct system text by taking parts before and after the <tools> block
                pre_tools_text = system_content_full_str[: tools_match.start()]
                post_tools_text = system_content_full_str[tools_match.end() :]
                system_text_for_granite = (pre_tools_text + post_tools_text).strip()

                try:
                    parsed_hermes_tools = ast.literal_eval(tools_str_from_hermes)
                    if isinstance(parsed_hermes_tools, dict):
                        parsed_hermes_tools = [parsed_hermes_tools]

                    if isinstance(parsed_hermes_tools, list) and all(
                        isinstance(item, dict) for item in parsed_hermes_tools
                    ):
                        tools_json_for_granite = json.dumps(parsed_hermes_tools)
                    else:
                        if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
                            eprint_rank0(
                                f"[WARNING] Example {example_id_for_log}: Parsed tools from <tools> tag is not a list of dicts. Content: {tools_str_from_hermes[:100]}. Using '[]'."
                            )
                        # tools_json_for_granite remains "[]"
                except (SyntaxError, ValueError) as e:
                    if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
                        eprint_rank0(
                            f"[WARNING] Example {example_id_for_log}: Failed to parse tools string: '{tools_str_from_hermes[:100]}...'. Error: {e}. Using '[]'."
                        )
                    # tools_json_for_granite remains "[]"

            if system_text_for_granite:
                formatted_text_parts.append(
                    format_granite_turn(ROLE_SYSTEM_GRANITE, system_text_for_granite)
                )

            formatted_text_parts.append(
                format_granite_turn(
                    ROLE_AVAILABLE_TOOLS_GRANITE, tools_json_for_granite
                )
            )

        elif role == ROLE_HUMAN_HERMES:
            formatted_text_parts.append(
                format_granite_turn(ROLE_USER_GRANITE, str(value).strip())
            )

        elif role == ROLE_GPT_HERMES:
            assistant_content_full = str(value).strip()
            tool_call_matches = list(
                re.finditer(
                    r"<tool_call>(.*?)</tool_call>",
                    assistant_content_full,
                    re.DOTALL | re.IGNORECASE,
                )
            )

            if tool_call_matches:
                granite_tool_calls = []
                processed_text_upto = 0
                final_assistant_response_parts = []

                for tc_match in tool_call_matches:
                    text_before_tc = assistant_content_full[
                        processed_text_upto : tc_match.start()
                    ].strip()
                    if text_before_tc:
                        final_assistant_response_parts.append(text_before_tc)
                    tool_call_str_hermes = tc_match.group(1).strip()
                    try:
                        tool_call_dict_hermes = ast.literal_eval(tool_call_str_hermes)
                        granite_tool_call_obj = {
                            "name": tool_call_dict_hermes.get("name")
                        }
                        args_hermes = tool_call_dict_hermes.get("arguments", {})
                        if isinstance(args_hermes, dict):
                            granite_tool_call_obj["arguments"] = json.dumps(args_hermes)
                        elif isinstance(args_hermes, str):
                            try:
                                json.loads(args_hermes)
                                granite_tool_call_obj["arguments"] = args_hermes
                            except json.JSONDecodeError:
                                if (
                                    int(os.getenv("RANK", "0")) == 0
                                    and example_id_for_log < 5
                                ):
                                    eprint_rank0(
                                        f"[WARNING] Example {example_id_for_log}: Assistant function_call arguments string is not valid JSON: {args_hermes[:200]}... Wrapping in quotes."
                                    )
                                granite_tool_call_obj["arguments"] = json.dumps(
                                    str(args_hermes)
                                )
                        else:
                            granite_tool_call_obj["arguments"] = "{}"
                        granite_tool_calls.append(granite_tool_call_obj)
                    except (SyntaxError, ValueError) as e:
                        if int(os.getenv("RANK", "0")) == 0 and example_id_for_log < 5:
                            eprint_rank0(
                                f"[WARNING] Example {example_id_for_log}: Failed to parse Hermes tool call string: {tool_call_str_hermes[:200]}... Error: {e}"
                            )
                        final_assistant_response_parts.append(
                            f"<tool_call>{tool_call_str_hermes}</tool_call>"
                        )
                    processed_text_upto = tc_match.end()

                text_after_last_tc = assistant_content_full[
                    processed_text_upto:
                ].strip()
                if text_after_last_tc:
                    final_assistant_response_parts.append(text_after_last_tc)

                assistant_response_for_granite = " ".join(
                    final_assistant_response_parts
                ).strip()
                if granite_tool_calls:
                    if assistant_response_for_granite:
                        assistant_response_for_granite += " "
                    assistant_response_for_granite += (
                        f"{TOOL_CALL_MARKER_GRANITE}{json.dumps(granite_tool_calls)}"
                    )

                if (
                    not assistant_response_for_granite.strip()
                    and not granite_tool_calls
                ):
                    assistant_response_for_granite = ""

                formatted_text_parts.append(
                    format_granite_turn(
                        ROLE_ASSISTANT_GRANITE, assistant_response_for_granite
                    )
                )
            else:
                formatted_text_parts.append(
                    format_granite_turn(ROLE_ASSISTANT_GRANITE, assistant_content_full)
                )

    final_formatted_text = "".join(formatted_text_parts)

    if not final_formatted_text.strip():
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
        description="Preprocess NousResearch/hermes-function-calling-v1 dataset for Granite and save to disk."
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
        default="NousResearch/hermes-function-calling-v1",
        help="Name of the dataset on Hugging Face Hub.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        default="../glaive_fc_v2",
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

    special_tokens_to_add_if_missing = {
        "pad_token": tokenizer.eos_token if tokenizer.eos_token else "[PAD]",
        "additional_special_tokens": [SOT, EOTR, EOTXT, TOOL_CALL_MARKER_GRANITE],
    }
    tokens_to_actually_add = []
    if (
        tokenizer.pad_token is None
        and special_tokens_to_add_if_missing["pad_token"] not in tokenizer.vocab
    ):
        tokens_to_actually_add.append(special_tokens_to_add_if_missing["pad_token"])
        tokenizer.pad_token = special_tokens_to_add_if_missing["pad_token"]
        print_rank0(f"Set tokenizer.pad_token to: {tokenizer.pad_token}")

    for token in special_tokens_to_add_if_missing["additional_special_tokens"]:
        if (
            token not in tokenizer.vocab
            and token not in tokenizer.get_added_vocab().values()
        ):
            tokens_to_actually_add.append(token)

    if tokens_to_actually_add:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens_to_actually_add}
        )
        print_rank0(f"Added special tokens to tokenizer: {tokens_to_actually_add}")

    print_rank0(f"Loading raw dataset: {script_args.dataset_name}")
    try:
        raw_dataset_train = load_dataset(script_args.dataset_name, split="train")
    except Exception as e:
        eprint_rank0(f"Failed to load dataset: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    print_rank0(f"Raw 'train' dataset loaded. Total examples: {len(raw_dataset_train)}")

    if "conversations" not in raw_dataset_train.column_names:
        eprint_rank0(
            f"[ERROR] Dataset {script_args.dataset_name} does not have a 'conversations' column."
        )
        sys.exit(1)

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
        remove_columns=[
            col for col in raw_dataset_subset.column_names if col != "idx_col"
        ],
        desc="Preprocessing Hermes dataset for Granite format",
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
        if "idx_col" in processed_dataset.column_names:
            processed_dataset = processed_dataset.remove_columns(["idx_col"])

        final_columns = ["input_ids", "attention_mask", "labels"]
        if not all(col in processed_dataset.column_names for col in final_columns):
            eprint_rank0(
                f"[ERROR] Not all expected columns ({final_columns}) are present. Found: {processed_dataset.column_names}. Aborting."
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
