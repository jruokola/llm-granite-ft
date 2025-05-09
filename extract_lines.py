def print_specific_lines(filepath, line_numbers):
    """
    Prints specific lines from a file.

    Args:
        filepath (str): The path to the file.
        line_numbers (list of int): A list of 1-indexed line numbers to print.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines_to_print = sorted(list(set(line_numbers)))  # Ensure unique and sorted
            current_line_to_find_idx = 0
            for i, line_content in enumerate(f):
                current_line_num = i + 1
                if (
                    current_line_to_find_idx < len(lines_to_print)
                    and current_line_num == lines_to_print[current_line_to_find_idx]
                ):
                    print(f"--- Line {current_line_num} (raw) ---")
                    print(
                        repr(line_content)
                    )  # Print with repr to see all characters including newlines
                    print(f"--- End Line {current_line_num} ---\n")
                    current_line_to_find_idx += 1

                if current_line_to_find_idx >= len(lines_to_print):
                    break

            if current_line_to_find_idx < len(lines_to_print):
                print(
                    f"Warning: Could not find all requested lines. Last line found was for request index {current_line_to_find_idx - 1}."
                )
                print(
                    f"Highest line number in file might be less than {lines_to_print[current_line_to_find_idx]}."
                )

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    FILE_TO_INSPECT = "strategic_game_chess.jsonl"
    LINES_TO_EXTRACT = [1, 6115]  # 1-indexed

    print(f"Inspecting file: {FILE_TO_INSPECT}")
    print(f"Attempting to extract lines: {LINES_TO_EXTRACT}\n")
    print_specific_lines(FILE_TO_INSPECT, LINES_TO_EXTRACT)
