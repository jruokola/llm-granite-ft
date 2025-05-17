import argparse
import json
import os
import random
import sys  # Re-add sys import

from datasets import Dataset, Features, Sequence, Value
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
            "input_ids": Sequence(feature=Value(dtype="int32")),
            "attention_mask": Sequence(feature=Value(dtype="int8")),
            "labels": Sequence(feature=Value(dtype="int64")),
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
    log_info("--- Synthetic Dataset Generation Complete ---")
