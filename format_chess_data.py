import json
import os


def fix_json_line(line_content):
    """
    Attempts to fix a line of JSON content, primarily by escaping unescaped newlines.
    Returns a tuple: (processed_line_content, was_fixed_or_originally_valid, error_message)
    Error_message is None if no error or if fix was successful.
    """
    try:
        # Try parsing the original line
        json.loads(line_content)
        return line_content, True, None  # Line is already valid
    except json.JSONDecodeError as e_orig:
        # Original line is not valid JSON. Attempt to fix unescaped newlines.
        corrected_content = line_content.replace("\n", "\\n")

        try:
            # Try parsing the corrected line
            json.loads(corrected_content)
            # If original was invalid but corrected is valid, it means fix worked.
            was_actually_fixed = corrected_content != line_content
            return corrected_content, True, None
        except json.JSONDecodeError as e_corrected:
            # The fix didn't work or introduced a new error.
            # Return original, failure, and the error from the corrected attempt,
            # or original error if content didn't change.
            if corrected_content != line_content:
                error_to_report = e_corrected
            else:
                error_to_report = e_orig
            return line_content, False, str(error_to_report)


def format_jsonl_file(input_filepath, output_filepath):
    """
    Reads a JSONL file, attempts to fix each line, and writes to an output file.
    """
    if not os.path.exists(input_filepath):
        print(f"Error: Input file not found at {input_filepath}")
        return False

    problematic_lines_count = 0
    processed_lines_count = 0
    fixed_lines_count = 0  # Counts lines that were changed and became valid
    originally_valid_count = 0  # Counts lines that were valid from the start

    with (
        open(input_filepath, "r", encoding="utf-8") as infile,
        open(output_filepath, "w", encoding="utf-8") as outfile,
    ):
        for i, line_str in enumerate(infile):
            processed_lines_count += 1
            original_line_content = line_str.rstrip("\r\n")

            fixed_line_content, success, error_msg = fix_json_line(
                original_line_content
            )

            if (
                success
            ):  # Only write if the line is valid (either originally or after fixing)
                outfile.write(fixed_line_content + "\n")
                if fixed_line_content != original_line_content:
                    fixed_lines_count += 1
                else:  # Success and content didn't change means it was originally valid
                    originally_valid_count += 1
            else:
                problematic_lines_count += 1
                if (i < 5) or (problematic_lines_count < 5):  # Print first few errors
                    print(
                        f"Warning: Line {i + 1} (original file) was invalid and could not be fixed. Error: {error_msg}. This line will be SKIPPED in the output."
                    )

    print("\nProcessing complete.")
    print(f"Total lines processed: {processed_lines_count}")
    print(f"Lines originally valid: {originally_valid_count}")
    print(f"Lines successfully fixed (and changed): {fixed_lines_count}")
    if problematic_lines_count > 0:
        print(
            f"Lines that were invalid and SKIPPED in the output: {problematic_lines_count}"
        )
    else:
        print("All lines processed successfully (either originally valid or fixed).")
    print(f"Formatted output written to: {output_filepath}")
    return True


if __name__ == "__main__":
    INPUT_FILE = "strategic_game_chess.jsonl"
    OUTPUT_FILE = "strategic_game_chess_formatted.jsonl"

    print(f"Starting JSONL formatting for: {INPUT_FILE}")
    print(f"Corrected output will be saved to: {OUTPUT_FILE}")

    if os.path.exists(INPUT_FILE):
        format_jsonl_file(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"Error: Input file '{INPUT_FILE}' not found in the current directory.")
        print(
            "Please ensure the file is present or update the INPUT_FILE path in the script."
        )
