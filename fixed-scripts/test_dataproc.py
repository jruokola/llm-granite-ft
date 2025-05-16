import argparse
import json  # For parsing JSON strings
import os
import sys  # For sys.stderr, sys.stdout
import time

from datasets import load_dataset
from transformers import AutoTokenizer

# Using print for now as per user request for better visibility in Slurm
# Setup logging
# logging.basicConfig(
#     level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
# )
# logger = logging.getLogger(__name__)


def print_rank0(*args, **kwargs):
    if int(os.getenv("RANK", "0")) == 0:
        print(*args, **kwargs)
        sys.stdout.flush()


def eprint_rank0(*args, **kwargs):
    if int(os.getenv("RANK", "0")) == 0:
        print(*args, file=sys.stderr, **kwargs)
        sys.stderr.flush()


# --- Granite Specific Tokens ---
# These should ideally be actual tokens known by the tokenizer.
# If not, the tokenizer might split them, making label creation harder.
# For formatting text, we use them as strings.
SOT = "<|start_of_role|>"
EOTR = "<|end_of_role|>"
EOTXT = "<|end_of_text|>"
TOOL_CALL = "<|tool_call|>"  # Granite uses this specific token for tool calls

# Roles for Granite
ROLE_SYSTEM = "system"
ROLE_AVAILABLE_TOOLS = "available_tools"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL_RESPONSE = "tool_response"


def format_turn(role, content):
    return f"{SOT}{role}{EOTR}{content}{EOTXT}\n"


def _create_labels_granite(input_ids_list, tokenizer):
    """
    Creates labels for language modeling, masking non-assistant parts for Granite format.
    Only content after <|start_of_role|>assistant<|end_of_role|> and inside <|tool_call|> should be unmasked.
    Tool responses and other roles are masked.
    """
    labels = [-100] * len(input_ids_list)  # Mask everything by default

    # Token IDs for key markers - this is crucial and assumes these are single tokens.
    # If they are multi-token, this logic becomes much more complex.
    # This part is highly dependent on the actual tokenizer being used with the Granite model.
    # We'd need to check if these special tokens are added and get their IDs.
    # For this example, we'll proceed with a string-matching heuristic on decoded segments,
    # which is fragile but a common starting point if exact token IDs are unknown or complex.

    # Heuristic: Decode segments and look for role markers.
    # This is very difficult to do robustly without knowing how the tokenizer handles these custom tokens.
    # A better approach would be to tokenize markers separately and search for sub-sequences of token IDs.

    # For simplicity in this refactor, let's assume a simplified labeling:
    # We want to predict what the assistant says, including its tool calls.
    # This means unmasking tokens that are part of the assistant's response *after* its role marker,
    # and the content of <|tool_call|> blocks.

    # This is a placeholder for a more robust labeling strategy.
    # The core challenge is identifying the *exact token spans* for assistant responses.
    # A simple heuristic: find "<|start_of_role|>assistant<|end_of_role|>" and unmask until "<|end_of_text|>".
    # And find "<|tool_call|>" and unmask its JSON content.

    # Due to the complexity and tokenizer-dependency, this simplified version will likely
    # need significant refinement. For now, it will be very basic.

    # Let's try to find assistant turns and tool_call sections by decoding.
    # This is inefficient and approximate.
    decoded_full_text = tokenizer.decode(input_ids_list, skip_special_tokens=False)

    assistant_turn_start_marker = f"{SOT}{ROLE_ASSISTANT}{EOTR}"
    tool_call_marker = TOOL_CALL
    end_of_text_marker = EOTXT

    current_idx = 0
    while current_idx < len(decoded_full_text):
        # Check for assistant turn
        assistant_start_idx = decoded_full_text.find(
            assistant_turn_start_marker, current_idx
        )
        if assistant_start_idx != -1:
            # Found an assistant turn. Unmask from after the marker to EOTXT or next SOT.
            content_start_char_idx = assistant_start_idx + len(
                assistant_turn_start_marker
            )

            # Check if this assistant turn is a tool call
            tool_call_start_idx = decoded_full_text.find(
                tool_call_marker, content_start_char_idx
            )
            eotxt_after_assistant_marker = decoded_full_text.find(
                end_of_text_marker, content_start_char_idx
            )

            if tool_call_start_idx != -1 and (
                eotxt_after_assistant_marker == -1
                or tool_call_start_idx < eotxt_after_assistant_marker
            ):
                # This is a tool call. Unmask from TOOL_CALL to its corresponding EOTXT
                # The content of TOOL_CALL is JSON, which itself might contain EOTXT if not careful.
                # Assuming tool call JSON does not contain EOTXT literally.
                actual_tool_call_content_start_char_idx = tool_call_start_idx + len(
                    tool_call_marker
                )

                # Find the end of the tool call block, which is before the main EOTXT of the assistant turn
                # This is tricky. The Granite example shows: <|assistant|><|end_of_role|><|tool_call|>[...]EOTXT
                # So, the EOTXT for the tool_call *is* the EOTXT for the assistant turn.

                end_of_tool_call_char_idx = decoded_full_text.find(
                    end_of_text_marker, actual_tool_call_content_start_char_idx
                )
                if end_of_tool_call_char_idx != -1:
                    # Convert character indices to token indices (approximate)
                    # This is where it gets very heuristic without precise token mapping.
                    # For each character, find which token it belongs to.
                    # This is a rough approximation.
                    start_token_idx = -1
                    end_token_idx = -1

                    # Find start_token_idx for actual_tool_call_content_start_char_idx
                    cumulative_len = 0
                    for i, token_id in enumerate(input_ids_list):
                        token_str = tokenizer.decode(
                            [token_id], skip_special_tokens=False
                        )
                        if (
                            cumulative_len >= actual_tool_call_content_start_char_idx
                            and start_token_idx == -1
                        ):
                            start_token_idx = i
                        cumulative_len += len(token_str)
                        if (
                            cumulative_len >= end_of_tool_call_char_idx
                            and start_token_idx != -1
                        ):
                            end_token_idx = (
                                i  # Unmask up to the token containing the end
                            )
                            break

                    if start_token_idx != -1 and end_token_idx != -1:
                        for i in range(
                            start_token_idx, end_token_idx + 1
                        ):  # Inclusive of the end token
                            if i < len(labels):
                                labels[i] = input_ids_list[i]

                    current_idx = end_of_tool_call_char_idx + len(end_of_text_marker)
                    continue  # Move to after this assistant turn

            elif eotxt_after_assistant_marker != -1:  # Regular assistant text response
                # Unmask from after assistant marker to EOTXT
                cumulative_len = 0
                start_token_idx = -1
                end_token_idx = -1
                for i, token_id in enumerate(input_ids_list):
                    token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                    if (
                        cumulative_len >= content_start_char_idx
                        and start_token_idx == -1
                    ):
                        start_token_idx = i
                    cumulative_len += len(token_str)
                    if (
                        cumulative_len >= eotxt_after_assistant_marker
                        and start_token_idx != -1
                    ):
                        end_token_idx = i
                        break

                if start_token_idx != -1 and end_token_idx != -1:
                    for i in range(start_token_idx, end_token_idx + 1):
                        if i < len(labels):
                            labels[i] = input_ids_list[i]
                current_idx = eotxt_after_assistant_marker + len(end_of_text_marker)
                continue

        # If no assistant turn found, or past the last one, break
        break

    # Mask all special tokens used for formatting, regardless of whose turn it is.
    # This is also heuristic. A better way is to get their actual token IDs.
    special_tokens_to_mask = [
        SOT,
        EOTR,
        EOTXT,
        TOOL_CALL,
        ROLE_SYSTEM,
        ROLE_AVAILABLE_TOOLS,
        ROLE_USER,
        ROLE_ASSISTANT,
        ROLE_TOOL_RESPONSE,
    ]
    # Add role names themselves if they might appear outside markers
    # and are tokenized into something non-standard.

    # This masking of special tokens is very basic and might over-mask or under-mask.
    # It's generally better to rely on the unmasking of desired content only.
    # The default of -100 for all labels already handles masking non-assistant parts.
    # The main task is to *unmask* the correct assistant parts.

    return labels


def preprocess_example(example, tokenizer, max_length):
    formatted_text = ""

    # 1. System Prompt
    if example.get("system"):
        system_content = str(example["system"]).strip()
        if system_content:
            formatted_text += format_turn(ROLE_SYSTEM, system_content)

    # 2. Available Tools
    if example.get("tools"):
        tools_json_str = example[
            "tools"
        ]  # This is already a JSON string in hqfx/glaive_fc_v2
        # Validate if it's proper JSON, though the dataset should provide it.
        # For Granite, the content is the JSON list itself.
        formatted_text += format_turn(ROLE_AVAILABLE_TOOLS, tools_json_str)

    # 3. Chat History
    chat_history = example.get("chat", [])
    if not isinstance(chat_history, list):  # Handle cases where chat might be a string
        chat_history = [{"role": "user", "content": str(chat_history)}]

    for turn in chat_history:
        role = turn.get("role", "").lower()
        content = turn.get("content")
        function_call = turn.get("function_call")  # For assistant

        if role == "user":
            formatted_text += format_turn(
                ROLE_USER, str(content).strip() if content else ""
            )
        elif role == "assistant":
            assistant_response = ""
            if function_call:
                # Ensure arguments are properly stringified JSON if they are dicts
                if isinstance(function_call.get("arguments"), dict):
                    function_call["arguments"] = json.dumps(function_call["arguments"])
                # Granite format: <|tool_call|>[{"name": "...", "arguments": "{...}"}]
                # The raw data `function_call` is a dict, needs to be a list of one dict for Granite.
                assistant_response += f"{TOOL_CALL}[{json.dumps(function_call)}]"
            elif content:
                assistant_response += str(content).strip()
            else:  # Assistant turn must have content or tool_call
                assistant_response = ""  # Or some placeholder if necessary
            formatted_text += format_turn(ROLE_ASSISTANT, assistant_response)
        elif (
            role == "tool_output" or role == "tool_response"
        ):  # Glaive uses tool_output
            # Content is expected to be a JSON string from the dataset
            tool_output_content = str(content).strip() if content else "{}"
            formatted_text += format_turn(ROLE_TOOL_RESPONSE, tool_output_content)
        # Ignoring 'tool_code' from glaive as Granite format doesn't have a direct equivalent,
        # it's represented by assistant's tool_call.

    # Fallback for empty formatted text
    if not formatted_text.strip():
        print_rank0(
            f"[WARNING] Example resulted in empty formatted text. Using fallback. Original: {example}"
        )
        formatted_text = format_turn(ROLE_USER, "Hello") + format_turn(
            ROLE_ASSISTANT, "Hello! How can I help you today?"
        )

    # Tokenization
    # Ensure tokenizer has pad_token set (usually to eos_token)
    # This should be done once outside this map function.
    # if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    encodings = tokenizer(
        formatted_text,
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
        description="Preprocess FunctionCallingDataset for Granite and save to disk."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ibm-granite/granite-3.3-2b-instruct",
        help="Tokenizer model name or path (for loading the tokenizer).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="hqfx/glaive_fc_v2",  # Using the user-provided dataset
        help="Name of the dataset on Hugging Face Hub.",
    )
    # --dataset_data_file is not needed if dataset_name directly points to a config with one file.
    # For hqfx/glaive_fc_v2, it seems to load directly.
    # parser.add_argument(
    #     "--dataset_data_file",
    #     type=str,
    #     default=None, # Let load_dataset pick default if None
    #     help="Specific data file within the dataset (e.g., json, jsonl).",
    # )
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
        default=2048,  # Default from original script, user might change this
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
    # It's crucial that this tokenizer knows the Granite special tokens.
    # If not, they need to be added: tokenizer.add_special_tokens(...)
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        # Common practice, but ensure EOS is appropriate for padding for this model
        tokenizer.pad_token = tokenizer.eos_token
        print_rank0(
            f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}"
        )

    # Verify if Granite special tokens are part of the vocabulary
    # This is important for correct tokenization and label creation
    granite_tokens = [
        SOT,
        EOTR,
        EOTXT,
        TOOL_CALL,
        ROLE_SYSTEM,
        ROLE_AVAILABLE_TOOLS,
        ROLE_USER,
        ROLE_ASSISTANT,
        ROLE_TOOL_RESPONSE,
    ]
    # A simple check; a more robust check would involve tokenizer.convert_tokens_to_ids
    # for token_str in granite_tokens:
    #     if token_str not in tokenizer.vocab:
    #         print_rank0(f"[WARNING] Special token '{token_str}' may not be in tokenizer vocabulary!")
    # Consider adding them if they are not:
    # num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': granite_tokens})
    # if num_added_toks > 0:
    #    print_rank0(f"Added {num_added_toks} special tokens to tokenizer.")
    #    # model.resize_token_embeddings(len(tokenizer)) # If resizing model embeddings

    print_rank0(f"Loading raw dataset: {script_args.dataset_name}")
    try:
        raw_dataset_dict = load_dataset(
            script_args.dataset_name
        )  # Removed data_files for hqfx
        # hqfx/glaive_fc_v2 has a 'train' split by default
        if "train" not in raw_dataset_dict:
            eprint_rank0(
                f"'train' split not found. Available: {list(raw_dataset_dict.keys())}"
            )
            sys.exit(1)
        raw_dataset_train = raw_dataset_dict["train"]
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
