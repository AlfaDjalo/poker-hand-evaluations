from db import DB, open_db
import time
import matplotlib.pyplot as plt
import numpy as np

def check_hand_id_map(db):
    """
    Helper function to test hand_id_map.
    """
    hand_id_map = db.get_hand_ids()
    print()
    print(f"hand_id_map contains {len(hand_id_map)} hands.")
    print(list(hand_id_map.items())[:5])
    print(":")
    print(list(hand_id_map.items())[-5:])
    print()

    return

def check_board_id_map(db):
    """
    Helper function to test board_id_map.
    """
    board_id_map = db.get_board_ids()
    print()
    print(f"board_id_map contains {len(board_id_map)} boards.")
    print(list(board_id_map.items())[:5])
    print(":")
    print(list(board_id_map.items())[-5:])
    print()

    for suit_pattern in range(5):
        board_id_map = db.get_board_ids(suit_pattern)
        print()
        print(f"board_id_map contains {len(board_id_map)} boards for suit_pattern {suit_pattern}.")
        print(list(board_id_map.items())[:5])
        print(":")
        print(list(board_id_map.items())[-5:])
        print()        

    return

def check_evaluations(db):
    """
    Helper function to test board_id_map.
    """
    start_time = time.time()

    # evaluations = db.get_evaluations_count()
    evaluations = db.get_evaluations_for_hand(42)
    evaluations = db.get_evaluations_count_for_hand(42)
    # evaluations = db.get_evaluations()

    end_time = time.time()

    print()
    print(f"evaluations contains {evaluations} evaluations.")
    print(f"Time taken: {end_time - start_time}")
    # print()
    # print(evaluations[:5])
    # print(":")
    # print(evaluations[-5:])
    # print()

    return


def check_evaluations_for_hand(db, hand_str):
    """
    Helper function to test board_id_map.
    """
    start_time = time.time()

    hand_id_map = db.get_hand_ids()
    hand_id = hand_id_map[hand_str]

    evaluations = db.get_evaluations_count_for_hand(hand_id)

    end_time = time.time()

    print()
    print(f"evaluations contains {evaluations} evaluations.")
    print(f"Time taken: {end_time - start_time}")
    # print()
    # print(evaluations[:5])
    # print(":")
    # print(evaluations[-5:])
    # print()

    return


def check_evaluations_for_hand(db, hand_str):
    """
    Helper function to test board_id_map.
    """
    start_time = time.time()

    hand_id_map = db.get_hand_ids()
    hand_id = hand_id_map[hand_str]

    evaluations = db.get_evaluations_count_for_hand(hand_id)

    end_time = time.time()

    print()
    print(f"evaluations contains {evaluations} evaluations.")
    print(f"Time taken: {end_time - start_time}")
    # print()
    # print(evaluations[:5])
    # print(":")
    # print(evaluations[-5:])
    # print()

    return


def plot_chart_for_hand(db, hand_str, ranking_type="rank_min"):

    hand_id_map = db.get_hand_ids()
    hand_id = hand_id_map[hand_str]

    evaluations = db.get_evaluations_for_hand(hand_id)

    # Get column names from cursor description
    col_names = [desc[0] for desc in db.cursor.description]

    if ranking_type not in col_names:
        raise ValueError(f"Ranking type '{ranking_type}' not found in table columns: {col_names}")

    # Find index of requested field
    field_idx = col_names.index(ranking_type)

    # Extract just the requested column from the rows
    values = [evaluation[field_idx] for evaluation in evaluations]

    # Plot histogram
    plt.hist(values, bins=50, edgecolor="black")
    plt.xlabel(ranking_type)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Hand Ranks for {hand_str}")
    plt.show()


def plot_rank_distribution(db, hand_str):
    start_time = time.time()

    # Get hand_id
    hand_id_map = db.get_hand_ids()
    hand_id = hand_id_map[hand_str]

    # Get all evaluations (full rows)
    rows = db.get_evaluations_for_hand(hand_id)

    # Get column indices for rank_min and rank_max
    col_names = [desc[0] for desc in db.cursor.description]
    rank_min_idx = col_names.index("rank_min")
    rank_max_idx = col_names.index("rank_max")

    # 100 bins from 0.00 to 1.00
    num_bins = 100
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_counts = np.zeros(num_bins)

    for row in rows:
        rmin = row[rank_min_idx]
        rmax = row[rank_max_idx]

        # Guard against zero-width ranges
        if rmax == rmin:
            # Put full weight into the single bin it falls into
            bin_idx = min(int(rmin * num_bins), num_bins - 1)
            bin_counts[bin_idx] += 1
            continue

        total_width = rmax - rmin

        # Find starting & ending bin indices
        start_bin = int(rmin * num_bins)
        # end_bin = int(rmax * num_bins)
        end_bin = int(min(rmax, 1.0 - 1e-12) * num_bins)
        
        for b in range(start_bin, end_bin + 1):
            bin_start = bin_edges[b]
            bin_end = bin_edges[b + 1]

            # Overlap between this bin and the [rmin, rmax] range
            overlap_start = max(bin_start, rmin)
            overlap_end = min(bin_end, rmax)
            overlap_width = max(0, overlap_end - overlap_start)

            if overlap_width > 0:
                bin_counts[b] += overlap_width / total_width  # Normalize so sum = 1 per row

    end_time = time.time()

    print(f"Time taken: {end_time - start_time}")


    # Plot histogram
    plt.bar(bin_edges[:-1] * 100, bin_counts, width=1.0, edgecolor="black", align="edge")
    plt.xlabel("Percentile (%)")
    plt.ylabel("Weighted Frequency")
    plt.title(f"Rank distribution for {hand_str}")
    plt.xlim(0, 100)
    plt.show()


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
    
    hand_id_map = db.get_hand_ids()

    for ax, hand_str in zip(axes, hand_strs):
        # Get hand_id
        hand_id = hand_id_map[hand_str]

        # Fetch all rows for this hand_id
        rows = db.get_evaluations_for_hand(hand_id)
        col_names = [desc[0] for desc in db.cursor.description]
        rank_min_idx = col_names.index("rank_min")
        rank_max_idx = col_names.index("rank_max")

        # Bin counts for this hand
        bin_counts = np.zeros(num_bins)

        for row in rows:
            rmin = row[rank_min_idx]
            rmax = row[rank_max_idx]

            # Handle zero-width ranges
            if rmax == rmin:
                bin_idx = min(int(rmin * num_bins), num_bins - 1)
                bin_counts[bin_idx] += 1
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
                    bin_counts[b] += overlap_width / total_width

        # Plot this hand's histogram
        ax.bar(bin_edges[:-1] * 100, bin_counts, width=1.0, edgecolor="black", align="edge")
        ax.set_ylabel("Weighted Freq")
        ax.set_title(f"Rank distribution for {hand_str}")
        ax.set_xlim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    axes[-1].set_xlabel("Percentile (%)")
    plt.tight_layout()
    plt.show()


def plot_rank_distribution_multi2(db, hand_strs):
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

        # Bin counts for this hand
        bin_counts = np.zeros(num_bins)

        for row in rows:
            rmin = row[rank_min_idx]
            rmax = row[rank_max_idx]

            # Handle zero-width ranges
            if rmax == rmin:
                bin_idx = min(int(rmin * num_bins), num_bins - 1)
                bin_counts[bin_idx] += 1
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
                    bin_counts[b] += overlap_width / total_width

        # Plot this hand's histogram
        ax.bar(bin_edges[:-1] * 100, bin_counts, width=1.0, edgecolor="black", align="edge")
        ax.set_ylabel("Weighted Freq")
        ax.set_title(f"Rank distribution for {hand_str}")
        ax.set_xlim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    axes[-1].set_xlabel("Percentile (%)")
    plt.tight_layout()
    plt.show()



def main():

    # Initialize DB connection
    db = open_db()

    # check_hand_id_map(db)
    # check_board_id_map(db)
    # check_evaluations(db)
    # check_evaluations_for_hand(db, "AhKd")
    # plot_chart_for_hand(db, "7h2c", "rank_min")
    # plot_rank_distribution(db, "7h7c")
    plot_rank_distribution_multi2(db, ["AA", "KK", "AKs", "AKo"])
    # Close the DB connection
    db.close()


if __name__ == "__main__":
    main()
