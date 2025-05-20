import argparse
import json
import os
import shutil
import sys

from datasets import Dataset, DatasetInfo, Features, Sequence, Value, load_dataset

# from mpire import WorkerPool # Temporarily removed for Dataset.from_generator refactor
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
ROLE_TOOL_RESPONSE_GRANITE = "tool_response"


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
        print_args = ("[WARNING]",) + args
        __builtins__.print(*print_args, file=sys.stderr, **kwargs)
        sys.stderr.flush()


def format_granite_turn(role, content):
    content_str = str(content).strip()
    return f"{SOT}{role}{EOTR}{content_str}{EOTXT}\n"


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


# Keep track of stats globally for the generator
stats = {
    "skipped_assistant_tool_call_turns": 0,
    "successfully_formatted_examples_count": 0,
    "skipped_examples_count": 0,
}


def _generate_processed_examples(
    source_dataset_iterable, tokenizer_obj, script_args_obj, system_prompt_str
):
    global stats  # Use global stats object
    stats["skipped_assistant_tool_call_turns"] = 0
    stats["successfully_formatted_examples_count"] = 0
    stats["skipped_examples_count"] = 0

    for i, raw_example in enumerate(source_dataset_iterable):
        if i % 1000 == 0 and i > 0:
            log_info(f"Formatting example {i}...")
        elif (
            script_args_obj.num_test_samples > 0
            and i < script_args_obj.num_test_samples
        ):
            log_info(f"Formatting example {i}...")

        example_parts = []
        current_example_had_issues = False

        example_parts.append(
            format_granite_turn(ROLE_SYSTEM_GRANITE, system_prompt_str)
        )

        system_field_content = raw_example.get("system", "")
        tools_marker = "Use them if required - "
        tools_json_str = "[]"
        if tools_marker in system_field_content:
            potential_tools_part = system_field_content.split(tools_marker, 1)[
                -1
            ].strip()
            extracted_tools = []
            decoder = json.JSONDecoder()
            idx = 0
            parse_tools_successful = True
            while idx < len(potential_tools_part):
                potential_tools_part = potential_tools_part[idx:].strip()
                if not potential_tools_part:
                    break
                try:
                    func, end_idx = decoder.raw_decode(potential_tools_part)
                    extracted_tools.append(func)
                    idx = end_idx
                except json.JSONDecodeError as e_tool_parse:
                    log_warning(
                        f"Could not parse all function JSONs in example {i} from: {potential_tools_part[idx : idx + 50]}... Error: {e_tool_parse}"
                    )
                    if not extracted_tools and potential_tools_part.strip() != "[]":
                        parse_tools_successful = False
                    break
            if extracted_tools:
                tools_json_str = json.dumps(extracted_tools)
            elif not parse_tools_successful:
                log_error(
                    f"Example {i}: Failed to extract any tools from system field, but tools seemed present. System field: {system_field_content[:200]}"
                )
        example_parts.append(
            format_granite_turn(ROLE_AVAILABLE_TOOLS_GRANITE, tools_json_str)
        )

        chat_content = raw_example.get("chat", "")
        if not chat_content:
            log_warning(f"Example {i}: Empty 'chat' field.")
            current_example_had_issues = True

        turns_raw = chat_content.split("<|endoftext|>")
        if (
            not any(turn_raw.strip() for turn_raw in turns_raw)
            and not current_example_had_issues
        ):
            log_warning(
                f"Example {i}: 'chat' field resulted in no valid turns after split by <|endoftext|>."
            )
            current_example_had_issues = True

        if (
            current_example_had_issues
        ):  # If chat is empty or yields no turns, skip example
            stats["skipped_examples_count"] += 1
            continue

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
                    fc_marker_start = "<functioncall>"
                    fc_json_str = content_part[len(fc_marker_start) :].strip()
                    try:
                        parsed_fc = json.loads(fc_json_str)
                        granite_fc_list_str = json.dumps([parsed_fc])
                        assistant_response_content = (
                            f"{TOOL_CALL_MARKER_GRANITE}{granite_fc_list_str}"
                        )
                        example_parts.append(
                            format_granite_turn(
                                ROLE_ASSISTANT_GRANITE, assistant_response_content
                            )
                        )
                    except json.JSONDecodeError as e:
                        log_error(
                            f"Failed to parse function call JSON in example {i} (ASSISTANT turn with tool_call skipped): {fc_json_str}. Error: {e}"
                        )
                        stats["skipped_assistant_tool_call_turns"] += 1
                else:
                    example_parts.append(
                        format_granite_turn(ROLE_ASSISTANT_GRANITE, content_part)
                    )
            elif turn_raw.startswith("FUNCTION RESPONSE:"):
                content = turn_raw[len("FUNCTION RESPONSE:") :].strip()
                example_parts.append(
                    format_granite_turn(ROLE_TOOL_RESPONSE_GRANITE, content)
                )
            else:
                log_warning(
                    f"Unrecognized turn structure in example {i}: {turn_raw[:100]}..."
                )

        text_content = "".join(example_parts)
        encodings = tokenizer_obj(
            text_content,
            max_length=script_args_obj.max_seq_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        labels = create_labels_for_granite_sequence(input_ids, tokenizer_obj)

        stats["successfully_formatted_examples_count"] += 1
        yield {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Download glaiveai/glaive-function-calling-v2, format it for Granite, and save to disk."
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
        default=4096,
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
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=-1,
        help="Number of samples to process for testing (default: -1, process all).",
    )
    # --num_workers is not used by Dataset.from_generator directly, but kept for potential future mpire re-integration inside generator
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (currently not used with from_generator).",
    )
    script_args = parser.parse_args()

    system_prompt = "Knowledge Cutoff Date: April 2024.\n Today's Date: April 12, 2025. You are Granite, developed by IBM. You are a helpful assistant with access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request."

    log_info(
        f"Loading source dataset: {script_args.hf_dataset_name}, split: {script_args.hf_dataset_split}"
    )
    try:
        # Consider stream=True for very large datasets if supported and beneficial
        source_dataset = load_dataset(
            script_args.hf_dataset_name, split=script_args.hf_dataset_split
        )
    except Exception as e:
        log_error(f"Failed to load dataset {script_args.hf_dataset_name}: {e}")
        sys.exit(1)
    log_info(f"Loaded {len(source_dataset)} examples from source.")

    if script_args.num_test_samples > 0:
        log_info(
            f"Processing only the first {script_args.num_test_samples} samples for testing."
        )
        source_dataset_iterable = source_dataset.select(
            range(min(script_args.num_test_samples, len(source_dataset)))
        )
    else:
        source_dataset_iterable = source_dataset
    log_info(f"Effective number of samples to process: {len(source_dataset_iterable)}")

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
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            log_info("Added [PAD] as pad_token.")

    final_features = Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int32"), length=-1),
            "attention_mask": Sequence(feature=Value(dtype="int8"), length=-1),
            "labels": Sequence(feature=Value(dtype="int64"), length=-1),
        }
    )

    log_info("Processing data using Dataset.from_generator...")
    # Pass necessary objects to the generator via gen_kwargs
    gen_kwargs = {
        "source_dataset_iterable": source_dataset_iterable,
        "tokenizer_obj": tokenizer,
        "script_args_obj": script_args,
        "system_prompt_str": system_prompt,
    }
    hf_dataset = Dataset.from_generator(
        _generate_processed_examples, features=final_features, gen_kwargs=gen_kwargs
    )

    log_info(
        f"Successfully formatted {stats['successfully_formatted_examples_count']} examples."
    )
    log_info(
        f"Skipped {stats['skipped_examples_count']} entire examples due to critical parsing issues."
    )
    if stats["skipped_assistant_tool_call_turns"] > 0:
        log_warning(
            f"Skipped {stats['skipped_assistant_tool_call_turns']} individual assistant tool call turns due to JSON parsing errors."
        )

    log_info(
        f"Saving processed dataset with {len(hf_dataset)} examples to {script_args.output_path}..."
    )
    hf_dataset.save_to_disk(script_args.output_path)

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
        )
        dataset_for_resave = dataset_loaded_after_patch.cast(final_features)
        log_info(
            f"Features from dataset after cast (pre-info update): {dataset_for_resave.features}"
        )

        if dataset_for_resave.info is not None:
            dataset_for_resave.info.features = final_features
            log_info(
                "Updated dataset_for_resave.info.features with known correct features."
            )
        else:
            log_info(
                "dataset_for_resave.info was None, creating new DatasetInfo object."
            )
            dataset_for_resave.info = DatasetInfo(features=final_features)

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
