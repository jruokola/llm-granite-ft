import json

# import torch # Not needed for this test script if only inspecting strings
import argparse
import logging
from transformers import AutoTokenizer

# Temporarily add llm-granite-ft to Python path to import JsonlDataset
# This is a common pattern for test scripts.
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the dataset class from your main script
# Assuming chess-finetune.py is in the same directory
from chess_finetune import JsonlDataset

# Setup basic logging for the test
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def create_dummy_datafile(filename="strategic_game_chess_test.jsonl"):
    # Content from the first line of the user's provided strategic_game_chess.jsonl
    first_line_content = {
        "text": "e2e4 c7c5 g1e2 g7g6 d2d4 f8g7 b1c3 g8f6 e4e5 f6g4 f2f4 c5d4 e2d4 g4h6 c1e3 d7d6 d1f3 b8c6 e1c1 c8g4 d4c6 d8d7 c6b8 a8b8 f3f2 e8g8 d1d2 b8c8 e3d4 h6f5 d4c5 d7e6 h2h3 c8c5 h3g4 c5c3 b2c3 e6a2 c1d1 f5h6 g4g5 h6g4 f2h4 h7h5 g5h6 g4h6 g2g4 d6e5 g4g5 a2a1 d1e2 a1a6 e2f2 a6b6 f2g2 b6c6 g2g1 c6c3 g5h6 c3d2 h6g7 d2d4 g1g2 d4e4 g2f2 e4d4 f2e2 d4e4 e2d2 g8g7 f4e5 f8d8 f1d3 e4h4 h1h4 d8d5 h4c4 b7b6 c4c7 d5e5 c7a7 f7f5 d3c4 g7f6 a7a6 e7e6 a6b6 g6g5 b6b3 f6e7 b3e3 e7d6 e3e5 d6e5 d2e3 e5d6 c4a6 d6d5 e3d3 d5d6 d3e2 d6d5 a6b7 d5c4 e2d2 g5g4 b7c8 c4d5 d2d3 g4g3 c8b7 d5c5 d3e2 c5c4 e2f3 c4c3 f3g3 c3c2 g3g2 e6e5 g2f3 c2d3 f3f2 d3d2 f2f3 e5e4 b7e4 d2c3 e4d3 c3d4 f3f4 d4d5 d3c4 d5d6 c4e2 d6c5 e2b5 c5d6 b5c4 d6c5 c4a6 c5b6 a6d3 b6a5 d3b5 a5b4 b5f1 b4c5 f1d3 c5b6 d3a6 b6c6 a6b5 c6c5 b5d3 c5d5 d3f1 d5e6 f1c4 e6e7 c4b5 e7d6 b5a6 d6d5 a6f1 d5d4 f1a6 d4c3 a6d3 c3b4 d3a6 b4a5 a6d3 a5a4 d3b5 a4a3 b5a4 a3b4 a4b5 b4b3 b5d3 b3b4 d3c4 b4a4 c4b5 a4a5 b5f1 a5b6 f1e2 b6a5 e2f1 a5b6 f1e2 b6c7 e2b5 c7c8 f4f5 1/2-1/2"
    }
    with open(filename, "w") as f:
        json.dump(first_line_content, f)
    logger.info(f"Created dummy data file: {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Test script for chess data preprocessing."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ibm-granite/granite-3.3-8b-instruct",  # Changed to Granite tokenizer
        help="Tokenizer model name or path.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Max sequence length for testing.",
    )

    args = parser.parse_args()

    logger.info(f"Loading tokenizer from: {args.model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        logger.error(
            "Please ensure you have internet access and the model name is correct."
        )
        logger.error(
            "You might need to log in to Hugging Face CLI: `huggingface-cli login`"
        )
        return

    dummy_file_path = create_dummy_datafile()

    logger.info("Initializing JsonlDataset with dummy data...")
    dataset = JsonlDataset(
        file_path=dummy_file_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    if not dataset.samples:
        logger.error(
            "No samples were processed by JsonlDataset. Check the dummy data or dataset logic."
        )
        return

    logger.info(f"Number of samples in dataset: {len(dataset)}")

    sample_idx = 0
    if len(dataset) > sample_idx:
        try:
            # JsonlDataset.__getitem__ returns torch tensors, ensure torch is available if not mocked
            # For this specific test, if torch isn't in the environment, we might hit an error here.
            # However, the user context implies a PyTorch environment.
            import torch  # ensure torch is imported for tensor operations in __getitem__

            processed_sample = dataset[sample_idx]
            input_ids = processed_sample["input_ids"]
            labels = processed_sample["labels"]
        except ImportError:
            logger.error(
                "PyTorch is not installed. Cannot get sample from dataset as it returns tensors."
            )
            logger.error("Please install PyTorch: pip install torch")
            return
        except Exception as e:
            logger.error(f"Error getting sample from dataset: {e}")
            return

        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)

        predict_ids = [label_id for label_id in labels if label_id != -100]
        decoded_labels = tokenizer.decode(predict_ids, skip_special_tokens=False)

        output_content = (
            f"--- Preprocessing Test Output for Sample {sample_idx} ---\\n\\n"
        )
        output_content += (
            f"Original Context: {dataset.samples[sample_idx]['context']}\\n"
        )
        output_content += (
            f"Original Target: {dataset.samples[sample_idx]['target']}\\n\\n"
        )

        output_content += f"Tokenized Input IDs:\\n{input_ids.tolist()}\\n\\n"
        output_content += (
            f"Decoded Input (what model sees):\\n'''{decoded_input}'''\\n\\n"
        )

        output_content += f"Tokenized Labels:\\n{labels.tolist()}\\n\\n"
        output_content += (
            f"Decoded Labels (what model predicts):\\n'''{decoded_labels}'''\\n"
        )

        print("\\n" + output_content)

        output_filename = "preprocess_output.txt"
        with open(output_filename, "w") as f:
            f.write(output_content)
        logger.info(f"Output saved to {output_filename}")

    else:
        logger.warning(
            f"Dataset has fewer than {sample_idx + 1} samples. Cannot display sample."
        )

    # Clean up the dummy file
    # os.remove(dummy_file_path)
    # logger.info(f"Removed dummy data file: {dummy_file_path}")
    # Commented out to allow inspection of the dummy file if needed.


if __name__ == "__main__":
    main()
