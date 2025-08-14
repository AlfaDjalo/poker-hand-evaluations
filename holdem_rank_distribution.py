import matplotlib.pyplot as plt
import numpy as np
import os
import json

from db import DB, open_db
from deck import RANKS

PATTERN_COUNTS = [4, 12, 12, 12, 4, 12]


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
    Create hand distribution plot for given hand.

    Arguments:
        db: The evaluations database.
        hand_str: The string representation of the hand.

    Returns:
        List of the percentile frequencies for the hand.
        Chart of the rank distribution for the hand.
    """

    num_bins = 100
    bin_edges = np.linspace(0, 1, num_bins + 1)

    fig, ax = plt.subplots(figsize=(8, 3))

    rows = db.get_evaluations_for_suitedness(hand_str)
    col_names = [desc[0] for desc in db.cursor.description]
    rank_min_idx = col_names.index("rank_min")
    rank_max_idx = col_names.index("rank_max")
    suit_multiplier_idx = col_names.index("suit_pattern")

    # Bin counts for this hand
    bin_counts = np.zeros(num_bins)

    for row in rows:
        rmin = row[rank_min_idx]
        rmax = row[rank_max_idx]
        suit_multiplier = PATTERN_COUNTS[int(row[suit_multiplier_idx])]

        # Handle zero-width ranges
        if rmax == rmin:
            bin_idx = min(int(rmin * num_bins), num_bins - 1)
            bin_counts[bin_idx] += suit_multiplier
            continue

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
        
    num_hands = sum(bin_counts)
    bin_counts_normalized = bin_counts / num_hands

    # Plot this hand's histogram
    ax.bar(bin_edges[:-1] * 100, bin_counts_normalized, width=1.0, edgecolor="black", align="edge")
    ax.set_ylabel("Frequency (normalized)")
    ax.set_title(f"Rank distribution for {hand_str}")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.20)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    ax.set_xlabel("Percentile (%)")
    plt.tight_layout()
    # plt.show()

    return bin_counts_normalized.tolist(), plt



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
        rank_min_idx = col_names.index("rank_min")
        rank_max_idx = col_names.index("rank_max")
        suit_multiplier_idx = col_names.index("suit_pattern")
        # print(f"Suit_multiplier_idx {suit_multiplier_idx}")

        # Bin counts for this hand
        bin_counts = np.zeros(num_bins)

        for row in rows:
            rmin = row[rank_min_idx]
            rmax = row[rank_max_idx]
            suit_multiplier = PATTERN_COUNTS[int(row[suit_multiplier_idx])]
            # print(f"Suit_multiplier {suit_multiplier}")

            # Handle zero-width ranges
            if rmax == rmin:
                bin_idx = min(int(rmin * num_bins), num_bins - 1)
                bin_counts[bin_idx] += suit_multiplier
                continue

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


def main():

    # Initialize DB connection
    db = open_db()

    create_rank_chart_data(db)
    # plot_rank_distribution(db, "32o")
    # plot_rank_distribution_multi(db, ["AQo"])
    # plot_rank_distribution_multi(db, ["AQo", "JTs", "99"])

    # Close the DB connection
    db.close()


if __name__ == "__main__":
    main()
