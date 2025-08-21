import matplotlib.pyplot as plt
import numpy as np
import os
import json
from itertools import combinations

from db_plo import DB_PLO, open_db
from deck import RANKS
from card import RANK_ORDER, SUIT_ORDER #Card, card_sort_key

def create_rank_chart_data(db):
    """
    Create hand rank chart data for all hands.
    Calls plot_rank_distribution to return the chart
    and bin data for each hand.

    Arguments:
        db: The evaluations database.
    """    
    output_dir = "chart"
    os.makedirs(output_dir, exist_ok=True)


    all_chart_data = {}

    # for hand_str in ["AQo", "JTs", "99"]:

    for idx1, card1 in enumerate(RANKS):
        for idx2, card2 in enumerate(RANKS):
            if idx1 > idx2:
                hand_str = f"{card1}{card2}s"
            elif idx2 > idx1:
                hand_str = f"{card2}{card1}o"
            else:
                hand_str = f"{card1}{card2}"

            chart_data, chart = plot_rank_distribution(db, hand_str) 

            # Store data in dictionary
            all_chart_data[hand_str] = chart_data

            # Save chart image
            chart_path = os.path.join(output_dir, f"{hand_str}.png")
            chart.savefig(chart_path, dpi=300, bbox_inches='tight')
            print(f"Saved chart for {hand_str} -> {chart_path}")


    # Save all chart data to JSON
    json_path = os.path.join(output_dir, "chart_data.json")
    with open(json_path, "w") as f:
        json.dump(all_chart_data, f, indent=4)
    print(f"Saved chart data -> {json_path}")

    return


def plot_rank_distribution(db, hand_str):
    """
    Create histogram of rank_dense values for given hand.

    Arguments:
        db: The evaluations database.
        hand_str: The string representation of the hand.

    Returns:
        Histogram counts (normalized) and the plot object.
    """

    # Fetch rows for this hand
    rows = db.get_evaluations_for_suitedness(hand_str)
    col_names = [desc[0] for desc in db.cursor.description]
    rank_dense_idx = col_names.index("rank_dense")

    # Extract the rank_dense column values
    rank_values = [row[rank_dense_idx] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 3))

    # Plot histogram directly
    counts, bins, patches = ax.hist(
        rank_values,
        bins=100,
        range=(0, 1),
        density=True,       # normalize to frequency distribution
        edgecolor="black"
    )

    ax.set_ylabel("Frequency (normalized)")
    ax.set_title(f"Rank distribution for {hand_str}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(counts) * 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    ax.set_xlabel("Rank (0–1)")
    plt.tight_layout()

    return counts.tolist(), plt


def plot_rank_distribution_multi(db, hand_strs):
    """
    Plots up to 4 hands in separate subplots.
    Each subplot shows the rank distribution from rank_min to rank_max,
    weighted so each evaluation contributes total weight = 1.
    """

    num_bins = 100
    bin_edges = np.linspace(0, 1, num_bins + 1)

    fig, axes = plt.subplots(len(hand_strs), 1, figsize=(8, 3 * len(hand_strs)), sharex=True)

    if len(hand_strs) == 1:
        axes = [axes]  # make iterable if only one subplot
    
    # hand_id_map = db.get_hand_ids()

    for ax, hand_str in zip(axes, hand_strs):
        # Get hand_id
        # hand_id = hand_id_map[hand_str]
        # hand_id = hand_strs[hand_str]

        # Fetch all rows for this hand_id
        rows = db.get_evaluations_for_suitedness(hand_str)
        col_names = [desc[0] for desc in db.cursor.description]
        rank_idx = col_names.index("rank_dense")

        # Bin counts for this hand
        bin_counts = np.zeros(num_bins)

        for row in rows:
            rank = row[rank_idx]

            # Handle zero-width ranges
            total_width = rmax - rmin
            start_bin = int(rmin * num_bins)
            end_bin = min(int(rmax * num_bins), num_bins - 1)

            for b in range(start_bin, end_bin + 1):
                bin_start = bin_edges[b]
                bin_end = bin_edges[b + 1]
                overlap_start = max(bin_start, rmin)
                overlap_end = min(bin_end, rmax)
                overlap_width = max(0, overlap_end - overlap_start)
                if overlap_width > 0:
                    bin_counts[b] += suit_multiplier * overlap_width / total_width
            # print(f"Total hands for {suit_multiplier} = {sum(bin_counts)}")
            
        num_hands = sum(bin_counts)
        # print(f"Total hands for {hand_str} = {sum(bin_counts)}")
        bin_counts_normalized = bin_counts / num_hands

        # Plot this hand's histogram
        ax.bar(bin_edges[:-1] * 100, bin_counts_normalized, width=1.0, edgecolor="black", align="edge")
        # ax.bar(bin_edges[:-1] * 100, bin_counts, width=1.0, edgecolor="black", align="edge")
        ax.set_ylabel("Frequency (normalized)")
        ax.set_title(f"Rank distribution for {hand_str}")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 0.1)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    axes[-1].set_xlabel("Percentile (%)")
    plt.tight_layout()
    plt.show()

def value_on_board(db, hand_arr, board_arr):
    """
    Calculates the best rank a hand can make on a given board.
    
    Arguments:
        db: The PLO evaluations database.
        hand_str: Array of string representing the hand being evaluated.
        board_str: Array of strings representing the cards for the board being evaluated.

    Returns:
        Tuple of (value, rank) for the hand.
    """
    hand_id_map = db.get_hand_ids()
    board_id_map = db.get_board_ids()
    lowest_value = float("inf")

    # Generate all 2-card hand combos (each sorted with card_sort_key)
    hand_combos = [
        "".join(sorted([c for c in combo], key=card_sort_key))
        for combo in combinations(hand_arr, 2)
    ]

    # Generate all 3-card board combos (each sorted with card_sort_key)
    board_cards = [c for c in board_arr]
    board_combos = [ "".join(sorted([c for c in combo], key=card_sort_key))
                     for combo in combinations(board_cards, 3) ]

    for hand in hand_combos:
        hand_id = hand_id_map.get(hand)
        if hand_id is None:
            continue

        hand_evaluations = db.get_evaluations_for_hand(hand_id)
        for board_key in board_combos:
            board_id = board_id_map.get(board_key)
            if board_id is None:
                continue

            evals_for_board = [row for row in hand_evaluations if row[0] == board_id]
            for _, _, value, rank in evals_for_board:
                print(f"Board: {board_key}, Hand: {hand}, Hand Value: {value}, Rank: {rank}")
                lowest_value = min(lowest_value, value)

    return lowest_value if lowest_value != float("inf") else None


def show_all_boards_with_card(db, card_str):
    """
    Prints all the boards containing a given card.
    
    Arguments:
        db: The PLO database.
        card_str: String representing the card.
        
    Returns:
        Nothing.
    """
    boards = db.get_boards_with_card(card_str)

    print(boards)

    return

def card_sort_key(card_str: str):
    """
    Sorting key for cards represented as 2-character strings like "As", "Qh", "Td".
    """
    rank = card_str[0]
    suit = card_str[1]
    return (RANK_ORDER[rank], SUIT_ORDER[suit])

def main():

    # Initialize DB connection
    db = open_db()

    # create_rank_chart_data(db)
    # chart_data, chart = plot_rank_distribution(db, "AA")
    # chart.show()
    # print(chart_data)
 
    # show_all_boards_with_card(db, "Tc")

    rank = value_on_board(db, ["As", "Kh", "Jc", "3c"], ["Ac", "Qd", "Jh", "Tc", "Qc"])
    print(rank)
 
    # Close the DB connection
    db.close()


if __name__ == "__main__":
    main()
