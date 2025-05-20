import argparse
import json
import os
import random
import shutil  # Added for directory operations
import sys  # Re-add sys import

from datasets import Dataset, DatasetInfo, Features, Sequence, Value
from transformers import AutoTokenizer

# --- Granite Specific Tokens & Roles (consistent with other scripts) ---
SOT = "<|start_of_role|>"
EOTR = "<|end_of_role|>"
EOTXT = "<|end_of_text|>"
TOOL_CALL_MARKER_GRANITE = "<|tool_call|>"

ROLE_SYSTEM_GRANITE = "system"
ROLE_AVAILABLE_TOOLS_GRANITE = "available_tools"


# Helper print functions (defined early)
def log_info(*args, **kwargs):  # Renamed from print
    # Simple print for non-distributed script, or for rank 0 if run in such context
    if int(os.getenv("RANK", "0")) == 0:
        # Call the built-in print function
        __builtins__.print(*args, **kwargs)
        if "file" not in kwargs or kwargs["file"] == sys.stdout:
            sys.stdout.flush()


def log_error(*args, **kwargs):  # Renamed from eprint
    if int(os.getenv("RANK", "0")) == 0:
        # Call the built-in print function
        __builtins__.print(*args, file=sys.stderr, **kwargs)
        sys.stderr.flush()


ROLE_USER_GRANITE = "user"
ROLE_ASSISTANT_GRANITE = "assistant"
ROLE_TOOL_RESPONSE_GRANITE = "tool_response"


def format_granite_turn(role, content):
    content_str = str(content).strip()
    return f"{SOT}{role}{EOTR}{content_str}{EOTXT}\n"


# Labeling function (similar to the one in test_dataproc.py)
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


def generate_examples(num_examples):
    examples = []

    system_prompt = "Knowledge Cutoff Date: April 2024.\n Today's Date: April 12, 2025. You are Granite, developed by IBM. You are a helpful assistant with access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request."

    tools = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        },
        {
            "name": "get_stock_price",
            "description": "Retrieves the current stock price for a given ticker symbol...",  # Truncated for brevity
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol, e.g. AAPL for Apple Inc.",
                    }
                },
                "required": ["ticker"],
            },
        },
    ]
    tools_json_str = json.dumps(tools, indent=4)

    for i in range(num_examples):
        example_parts = []
        example_parts.append(format_granite_turn(ROLE_SYSTEM_GRANITE, system_prompt))
        example_parts.append(
            format_granite_turn(ROLE_AVAILABLE_TOOLS_GRANITE, tools_json_str)
        )

        # Alternate between weather and stock, and direct answer vs tool_response
        if i % 4 == 0:  # Weather query, direct tool call
            user_query = f"What's the weather like in London today {i}?"
            example_parts.append(format_granite_turn(ROLE_USER_GRANITE, user_query))
            tool_call_args = {"location": "London, UK"}
            tool_call_obj = [
                {"name": "get_current_weather", "arguments": json.dumps(tool_call_args)}
            ]
            assistant_response = (
                f"{TOOL_CALL_MARKER_GRANITE}{json.dumps(tool_call_obj)}"
            )
            example_parts.append(
                format_granite_turn(ROLE_ASSISTANT_GRANITE, assistant_response)
            )
            tool_response_content = json.dumps(
                {
                    "temperature": f"{20 + i % 5}.{i % 10}",
                    "unit": "C",
                    "condition": random.choice(["Sunny", "Cloudy", "Rainy"]),
                }
            )
            example_parts.append(
                format_granite_turn(ROLE_TOOL_RESPONSE_GRANITE, tool_response_content)
            )
            # Optional final assistant summary - for simplicity, not adding for all tool calls now
            # example_parts.append(format_granite_turn(ROLE_ASSISTANT_GRANITE, f"The weather in London is {tool_response_content}."))

        elif i % 4 == 1:  # Stock query, direct tool call
            tickers = ["IBM", "AAPL", "MSFT", "GOOG"]
            ticker = tickers[i % len(tickers)]
            user_query = f"Can you get me the stock price for {ticker}?"
            example_parts.append(format_granite_turn(ROLE_USER_GRANITE, user_query))
            tool_call_args = {"ticker": ticker}
            tool_call_obj = [
                {"name": "get_stock_price", "arguments": json.dumps(tool_call_args)}
            ]
            assistant_response = (
                f"{TOOL_CALL_MARKER_GRANITE}{json.dumps(tool_call_obj)}"
            )
            example_parts.append(
                format_granite_turn(ROLE_ASSISTANT_GRANITE, assistant_response)
            )
            tool_response_content = json.dumps(
                {
                    "ticker": ticker,
                    "price": f"{150 + i % 20}.{i % 100:02d}",
                    "currency": "USD",
                }
            )
            example_parts.append(
                format_granite_turn(ROLE_TOOL_RESPONSE_GRANITE, tool_response_content)
            )
            # example_parts.append(format_granite_turn(ROLE_ASSISTANT_GRANITE, f"The current price for {ticker} is ${json.loads(tool_response_content)['price']} USD."))

        elif i % 4 == 2:  # User query not requiring a tool
            user_query = f"What is the capital of France {i}?"
            example_parts.append(format_granite_turn(ROLE_USER_GRANITE, user_query))
            assistant_response = "The capital of France is Paris."
            example_parts.append(
                format_granite_turn(ROLE_ASSISTANT_GRANITE, assistant_response)
            )

        else:  # User query that could use a tool, but assistant says it cannot perform
            user_query = f"Can you order a pizza for me to address 123 Main St {i}?"
            example_parts.append(format_granite_turn(ROLE_USER_GRANITE, user_query))
            assistant_response = "I'm sorry, I cannot order a pizza. I can only get weather information or stock prices."
            example_parts.append(
                format_granite_turn(ROLE_ASSISTANT_GRANITE, assistant_response)
            )

        examples.append("".join(example_parts))

    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic Granite function calling examples and save to disk."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the processed dataset.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=25,
        help="Number of synthetic examples to generate.",
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
        default=512,
        help="Maximum sequence length for tokenization.",
    )  # Shorter default for synthetic data

    script_args = parser.parse_args()

    log_info(f"Generating {script_args.num_examples} synthetic examples...")
    text_examples = generate_examples(script_args.num_examples)

    log_info(f"Loading tokenizer: {script_args.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name_or_path, trust_remote_code=True
    )

    # IMPORTANT: Ensure Granite special tokens are added to the tokenizer if not already present.
    # This script assumes they are part of the tokenizer's vocabulary.
    # Example: tokenizer.add_special_tokens({'additional_special_tokens': [SOT, EOTR, EOTXT, TOOL_CALL_MARKER_GRANITE]})
    # And model embeddings would need resizing in the training script.

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            log_info(
                f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}"
            )
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Fallback
            log_info("Added [PAD] as pad_token.")

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    log_info("Tokenizing and creating labels for generated examples...")
    for i, text_content in enumerate(text_examples):
        if int(os.getenv("RANK", "0")) == 0 and i < 2:  # Print first 2 formatted texts
            log_info(f"\n--- Generated Example {i + 1} Text (to be tokenized) ---")
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

    # Define features for the dataset
    features = Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int32"), length=-1),
            "attention_mask": Sequence(feature=Value(dtype="int8"), length=-1),
            "labels": Sequence(feature=Value(dtype="int64"), length=-1),
        }
    )

    # Create Hugging Face Dataset
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

    # --- Post-process dataset_info.json to ensure 'length' is present for Sequence features ---
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
                for feature_name, feature_def in dataset_info_content[
                    "features"
                ].items():
                    if (
                        isinstance(feature_def, dict)
                        and feature_def.get("_type") == "Sequence"
                    ):
                        if "length" not in feature_def:
                            log_info(
                                f"Adding missing 'length': -1 to Sequence feature '{feature_name}' in {dataset_info_path}"
                            )
                            feature_def["length"] = -1
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
    # --- End of post-processing ---

    # --- Attempt to fix Arrow metadata by reloading with patched info and re-saving ---
    log_info(
        f"Attempting to fix Arrow metadata in {script_args.output_path} by reloading and re-saving..."
    )
    try:
        # Load the dataset. This will use the patched dataset_info.json.
        dataset_loaded_after_patch = Dataset.load_from_disk(script_args.output_path)
        log_info(
            f"Successfully reloaded dataset from {script_args.output_path} after patching dataset_info.json."
        )
        log_info(
            f"Features from reloaded dataset (before cast): {dataset_loaded_after_patch.features}"
        )

        # Explicitly cast to the correct 'features' object (defined in the main script scope)
        # This ensures the in-memory representation is correct before re-saving.
        log_info(f"Casting reloaded dataset to known correct features: {features}")
        dataset_for_resave = dataset_loaded_after_patch.cast(features)
        log_info(
            f"Features from dataset after cast (pre-info update): {dataset_for_resave.features}"
        )

        # Explicitly update the .info.features attribute of the dataset to be re-saved
        if dataset_for_resave.info is not None:
            dataset_for_resave.info.features = features
            log_info(
                "Updated dataset_for_resave.info.features with known correct features."
            )
        else:
            # This case is less likely for a loaded dataset but handle defensively
            log_info(
                "dataset_for_resave.info was None, creating new DatasetInfo object."
            )
            # from datasets import DatasetInfo # Ensure DatasetInfo is imported (already handled at top of file)
            dataset_for_resave.info = DatasetInfo(features=features)

        log_info(
            f"Features from dataset after cast and info update (to be re-saved): {dataset_for_resave.features}"
        )
        if dataset_for_resave.info:
            log_info(
                f"DatasetInfo.features after cast and info update: {dataset_for_resave.info.features}"
            )

        # Create a temporary directory for the re-save
        temp_resave_path = script_args.output_path + "__resaved_temp"
        if os.path.exists(temp_resave_path):
            shutil.rmtree(
                temp_resave_path
            )  # Clean up if exists from a previous failed run
        os.makedirs(temp_resave_path, exist_ok=True)

        log_info(f"Re-saving dataset to temporary path: {temp_resave_path}")
        dataset_for_resave.save_to_disk(temp_resave_path)
        log_info(
            f"Dataset re-saved to {temp_resave_path}. Now replacing original files."
        )

        # Replace original files with the re-saved ones.
        # This ensures the main script uses the dataset with potentially fixed Arrow metadata.
        original_data_files = [
            f
            for f in os.listdir(script_args.output_path)
            if os.path.isfile(os.path.join(script_args.output_path, f))
        ]

        for item_name in os.listdir(temp_resave_path):
            source_item_path = os.path.join(temp_resave_path, item_name)
            destination_item_path = os.path.join(script_args.output_path, item_name)

            if os.path.isfile(source_item_path):
                if os.path.exists(destination_item_path) and os.path.isdir(
                    destination_item_path
                ):
                    shutil.rmtree(
                        destination_item_path
                    )  # Remove directory if it's in the way of a file
                elif os.path.exists(destination_item_path) and os.path.isfile(
                    destination_item_path
                ):
                    os.remove(destination_item_path)  # Remove file if it's in the way
                shutil.copy2(source_item_path, destination_item_path)
            # Note: save_to_disk typically doesn't create subdirectories for simple datasets.
            # If it did, shutil.copytree would be needed for directories.

        # Remove any original files that are not in the re-saved version (e.g. old .arrow files if naming changed)
        # This is a bit risky if save_to_disk changes file naming conventions, but usually, it's consistent.
        # For now, let's assume direct replacement of dataset_info.json, state.json, and the .arrow file(s) is sufficient.
        # A more robust cleanup might be needed if save_to_disk produces a different set of files.

        shutil.rmtree(temp_resave_path)  # Clean up temporary directory
        log_info(
            f"Original dataset at {script_args.output_path} updated with re-saved version. Arrow metadata should now be fixed."
        )

    except Exception as e:
        log_error(f"Error during Arrow metadata fix attempt (reload and re-save): {e}")
        log_error(
            f"The dataset at {script_args.output_path} might still have problematic Arrow metadata."
        )
    # --- End of Arrow metadata fix attempt ---

    log_info("--- Synthetic Dataset Generation Complete ---")
