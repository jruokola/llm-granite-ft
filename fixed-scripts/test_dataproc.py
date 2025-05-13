import argparse
import logging
import time

from datasets import load_dataset
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class FunctionCallingDatasetTester:
    def __init__(self, dataset, tokenizer, max_length=2048, subset_size=-1):
        self.tokenizer = tokenizer
        if subset_size > 0:
            self.dataset = dataset.select(range(min(subset_size, len(dataset))))
        else:
            self.dataset = dataset

        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if len(self.dataset) > 0:
            logger.info(f"Dataset sample keys: {list(self.dataset[0].keys())}")
            sample_chat = self.dataset[0].get(
                "chat", self.dataset[0].get("text", str(self.dataset[0]))
            )
            logger.info(
                f"Dataset sample first example content (first 200 chars): {sample_chat[:200] if isinstance(sample_chat, str) else str(sample_chat)[:200]}"
            )

    def __len__(self):
        return len(self.dataset)

    def safe_process_message(self, message, idx=None):
        """Process a single message safely without causing attribute errors."""
        try:
            if isinstance(message, dict):
                role = str(message.get("role", "")).lower()
                content = str(message.get("content", ""))
                if role == "system":
                    return f"<|system|>\n{content}\n"
                elif role == "user":
                    return f"<|user|>\n{content}\n"
                elif role == "assistant":
                    return f"<|assistant|>\n{content}\n"
                else:
                    return f"<|user|>\n{content}\n"  # Default to user
            elif isinstance(message, str):
                return f"<|user|>\n{message}\n"
            elif isinstance(message, list):
                result = ""
                for submessage in message:
                    result += self.safe_process_message(submessage, idx)
                return result
            else:
                if (
                    idx is not None and idx < 3
                ):  # Log only for first few problematic ones
                    logger.warning(
                        f"Unknown message type: {type(message)} for example {idx}. Content: {message}"
                    )
                return f"<|user|>\n{str(message)}\n"
        except Exception as e:
            if idx is not None and idx < 3:
                logger.error(
                    f"Error processing message for example {idx}: {str(e)}. Content: {message}"
                )
            return ""

    def __getitem__(self, idx):
        start_time = time.time()
        raw_example_content = ""
        formatted_text_len = 0
        input_ids_len = 0
        formatted_text_snippet = ""

        try:
            example = self.dataset[idx]
            formatted_text = ""

            if "system" in example and example["system"] is not None:
                system_content = str(example["system"]).strip()
                if system_content:
                    formatted_text += f"<|system|>\n{system_content}\n"

            chat_content = None
            if "chat" in example and example["chat"] is not None:
                chat_content = example["chat"]
            elif "text" in example and example["text"] is not None:
                chat_content = example["text"]
            else:
                chat_content = str(example)  # Fallback

            raw_example_content = str(chat_content)

            if isinstance(chat_content, str):
                lines = chat_content.split("\n")
                current_role = None
                current_content = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    new_role_detected = False
                    if line.startswith("SYSTEM:"):
                        current_role, current_content, formatted_text = (
                            self._flush_content_to_formatted_text(
                                current_role,
                                current_content,
                                formatted_text,
                                "system",
                                line.replace("SYSTEM:", "").strip(),
                            )
                        )
                        new_role_detected = True
                    elif line.startswith("USER:"):
                        current_role, current_content, formatted_text = (
                            self._flush_content_to_formatted_text(
                                current_role,
                                current_content,
                                formatted_text,
                                "user",
                                line.replace("USER:", "").strip(),
                            )
                        )
                        new_role_detected = True
                    elif line.startswith("A:"):
                        current_role, current_content, formatted_text = (
                            self._flush_content_to_formatted_text(
                                current_role,
                                current_content,
                                formatted_text,
                                "assistant",
                                line.replace("A:", "").strip(),
                            )
                        )
                        new_role_detected = True
                    elif line.startswith("FUNCTION RESPONSE:"):
                        if current_role == "assistant":
                            current_content.append(line)
                        else:  # If not in assistant, start new assistant message
                            current_role, current_content, formatted_text = (
                                self._flush_content_to_formatted_text(
                                    current_role,
                                    current_content,
                                    formatted_text,
                                    "assistant",
                                    line,
                                )
                            )
                        new_role_detected = True  # Treat as role boundary for logic

                    if not new_role_detected:
                        if current_role:
                            current_content.append(line)
                        else:  # Default to system if no role marker seen yet
                            current_role = "system"
                            current_content = [line]

                # Flush any remaining content
                _, _, formatted_text = self._flush_content_to_formatted_text(
                    current_role,
                    current_content,
                    formatted_text,
                    None,
                    None,
                    force_flush=True,
                )

            elif isinstance(chat_content, list):
                for message in chat_content:
                    formatted_text += self.safe_process_message(message, idx)
            else:  # Unknown type
                formatted_text += f"<|user|>\n{str(chat_content)}\n"

            formatted_text = formatted_text.replace("<|endoftext|>", "")
            if not formatted_text.strip():
                formatted_text = (
                    "<|user|>\nHello\n<|assistant|>\nHello! How can I help you today?\n"
                )

            formatted_text_len = len(formatted_text)
            formatted_text_snippet = formatted_text[:200].replace("\n", "\\n") + "..."

            # Tokenization (not returning PyTorch tensors)
            # Using return_tensors=None (default) or "np" would avoid torch dependency here.
            # For simplicity, let's assume the tokenizer can handle it or this part might be slow.
            encodings = self.tokenizer(
                formatted_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",  # This might be slow if not needed for pure speed test of processing
                return_attention_mask=True,  # Ensure attention_mask is returned
            )
            input_ids = encodings["input_ids"]
            # attention_mask = encodings["attention_mask"] # Available if needed
            input_ids_len = len(input_ids)

            processing_time = time.time() - start_time
            return {
                "idx": idx,
                "processing_time_seconds": processing_time,
                "raw_content_len": len(raw_example_content),
                "formatted_text_len": formatted_text_len,
                "formatted_text_snippet": formatted_text_snippet,
                "input_ids_len": input_ids_len,
                "error": None,
            }

        except Exception as e:
            import traceback

            logger.error(
                f"Critical error in __getitem__ for example {idx}: {str(e)}\n{traceback.format_exc()}"
            )
            processing_time = time.time() - start_time
            return {
                "idx": idx,
                "processing_time_seconds": processing_time,
                "raw_content_len": len(
                    raw_example_content
                ),  # Might be 0 if error early
                "formatted_text_len": formatted_text_len,
                "formatted_text_snippet": "ERROR DURING PROCESSING",
                "input_ids_len": input_ids_len,
                "error": str(e),
            }

    def _flush_content_to_formatted_text(
        self,
        current_role,
        current_content_list,
        formatted_text,
        next_role=None,
        next_content_line=None,
        force_flush=False,
    ):
        if current_role and current_content_list:
            role_content = "\n".join(current_content_list).strip()
            if role_content:  # Only add if there's actual content
                if current_role == "system":
                    # Add system message only if it's not already the start of formatted_text
                    # This check might be too simplistic if system messages can appear later.
                    # For now, assuming one leading system message or system messages are distinctly marked.
                    if not formatted_text.startswith("<|system|>"):
                        formatted_text += f"<|system|>\n{role_content}\n"
                    # If it is already there, and this is a new system block, it implies multiple system messages.
                    # The current logic might merge them or add duplicates if not careful.
                    # For simplicity, let's assume distinct blocks are fine.
                    # else: # formatted_text.startswith("<|system|>")
                    #    formatted_text += f"<|system|>\n{role_content}\n" # Allow multiple system blocks if needed
                elif current_role == "user":
                    formatted_text += f"<|user|>\n{role_content}\n"
                elif current_role == "assistant":
                    formatted_text += f"<|assistant|>\n{role_content}\n"

        if force_flush:
            return None, [], formatted_text  # Reset role and content

        # Start new role and content
        new_role = next_role
        new_content_list = [next_content_line.strip()] if next_content_line else []
        return new_role, new_content_list, formatted_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test data processing for FunctionCallingDataset."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ibm-granite/granite-3.3-2b-instruct",
        help="Tokenizer model name or path.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to test from the dataset.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization.",
    )
    script_args = parser.parse_args()

    logger.info(f"Loading tokenizer: {script_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, trust_remote_code=True
    )

    logger.info("Loading dataset: glaiveai/glaive-function-calling-v2")
    # Load only the 'train' split as that's what the original script uses primarily
    try:
        raw_dataset_full = load_dataset("glaiveai/glaive-function-calling-v2")
        # Assuming 'train' split exists, if not, this will error.
        # The original script splits this further, but for testing __getitem__ on raw data,
        # using a portion of the 'train' split directly is fine.
        raw_dataset = raw_dataset_full["train"]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        exit(1)

    logger.info(f"Dataset loaded. Total examples in 'train' split: {len(raw_dataset)}")

    # Use only a subset for testing if num_samples is less than dataset size
    num_to_process = min(script_args.num_samples, len(raw_dataset))
    if num_to_process < script_args.num_samples:
        logger.info(
            f"Requested {script_args.num_samples} samples, but dataset split has only {len(raw_dataset)}. Processing {num_to_process} samples."
        )

    dataset_tester = FunctionCallingDatasetTester(
        raw_dataset,  # Pass the selected split
        tokenizer,
        max_length=script_args.max_seq_length,
        subset_size=num_to_process,  # Process only the requested number of samples
    )

    # Corrected loop to use the length of the dataset_tester instance
    # which already considers the subset_size
    actual_samples_to_test = len(dataset_tester)
    logger.info(f"Processing {actual_samples_to_test} samples...")

    total_time = 0
    for i in range(actual_samples_to_test):
        logger.info(f"--- Processing sample index: {i} ---")
        metrics = dataset_tester[i]
        total_time += metrics["processing_time_seconds"]
        logger.info(f"  Time taken: {metrics['processing_time_seconds']:.4f} seconds")
        logger.info(f"  Raw content length: {metrics['raw_content_len']}")
        logger.info(f"  Formatted text length: {metrics['formatted_text_len']}")
        logger.info(f"  Formatted text snippet: {metrics['formatted_text_snippet']}")
        logger.info(f"  Input IDs length: {metrics['input_ids_len']}")
        if metrics["error"]:
            logger.error(f"  Error for sample {i}: {metrics['error']}")

    if actual_samples_to_test > 0:
        logger.info("--- Summary ---")
        logger.info(f"Processed {actual_samples_to_test} samples.")
        logger.info(f"Total processing time: {total_time:.4f} seconds.")
        logger.info(
            f"Average time per sample: {total_time / actual_samples_to_test:.4f} seconds."
        )
    else:
        logger.info("No samples were processed.")
