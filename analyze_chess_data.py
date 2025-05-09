import json
import os


def analyze_jsonl_data(file_path):
    """
    Analyzes a JSONL file containing chess game data.

    Each line is expected to be a JSON object with a "text" field
    containing the game moves as a string.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        dict: A dictionary containing statistics about the dataset.
              Returns None if the file is not found.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    num_games = 0
    total_text_length = 0
    min_text_length = float("inf")
    max_text_length = float("-inf")

    total_moves = 0
    min_moves = float("inf")
    max_moves = float("-inf")

    game_results_to_exclude = {"1-0", "0-1", "1/2-1/2"}

    with open(file_path, "r") as f:
        for line in f:
            try:
                game_data = json.loads(line)
                if "text" not in game_data:
                    print(
                        f"Warning: Skipping line due to missing 'text' field: {line.strip()}"
                    )
                    continue

                num_games += 1
                text = game_data["text"]
                current_text_length = len(text)

                total_text_length += current_text_length
                min_text_length = min(min_text_length, current_text_length)
                max_text_length = max(max_text_length, current_text_length)

                # Approximate move count
                all_tokens = text.split(" ")
                actual_moves_list = []
                for token in all_tokens:
                    if token not in game_results_to_exclude:
                        # Further filtering could be added here if move numbers (e.g., "1.") are present
                        # and should not be counted as moves. For now, any non-result token is a move.
                        if (
                            not token.endswith(".") or not token[:-1].isdigit()
                        ):  # basic filter for move numbers like "1."
                            actual_moves_list.append(token)
                    else:
                        break  # Stop if a game result is encountered

                current_num_moves = len(actual_moves_list)

                if (
                    current_num_moves > 0
                ):  # Avoid division by zero or issues with empty move lists for stats
                    total_moves += current_num_moves
                    min_moves = min(min_moves, current_num_moves)
                    max_moves = max(max_moves, current_num_moves)
                elif (
                    num_games == 1 and current_num_moves == 0
                ):  # Handle case where first game has 0 moves for min_moves
                    min_moves = 0

            except json.JSONDecodeError:
                print(
                    f"Warning: Skipping line due to JSON decode error: {line.strip()}"
                )
                continue

    if num_games == 0:
        print("No valid game data found in the file.")
        return {
            "num_games": 0,
            "avg_text_length": 0,
            "min_text_length": 0,
            "max_text_length": 0,
            "avg_moves_per_game": 0,
            "min_moves_per_game": 0,
            "max_moves_per_game": 0,
        }

    stats = {
        "num_games": num_games,
        "avg_text_length": total_text_length / num_games if num_games > 0 else 0,
        "min_text_length": min_text_length if min_text_length != float("inf") else 0,
        "max_text_length": max_text_length if max_text_length != float("-inf") else 0,
        "avg_moves_per_game": total_moves / num_games
        if num_games > 0
        else 0,  # Only consider games with moves for avg
        "min_moves_per_game": min_moves if min_moves != float("inf") else 0,
        "max_moves_per_game": max_moves if max_moves != float("-inf") else 0,
    }

    return stats


if __name__ == "__main__":
    file_to_analyze = "strategic_game_chess.jsonl"
    print(f"Analyzing {file_to_analyze}...")
    statistics = analyze_jsonl_data(file_to_analyze)

    if statistics:
        print("\\n--- Dataset Statistics ---")
        print(f"Total number of games: {statistics['num_games']:,}")
        print("Text length per game (characters):")
        print(f"  Average: {statistics['avg_text_length']:.2f}")
        print(f"  Min:     {statistics['min_text_length']:,}")
        print(f"  Max:     {statistics['max_text_length']:,}")
        print("Moves per game (approximate):")
        print(f"  Average: {statistics['avg_moves_per_game']:.2f}")
        print(f"  Min:     {statistics['min_moves_per_game']:,}")
        print(f"  Max:     {statistics['max_moves_per_game']:,}")
        print("--- End of Statistics ---")
