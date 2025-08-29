import matplotlib.pyplot as plt
import numpy as np
import os
import json
from itertools import combinations
import time

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
                # print(f"Board: {board_key}, Hand: {hand}, Hand Value: {value}, Rank: {rank}")
                lowest_value = min(lowest_value, value)

    return lowest_value if lowest_value != float("inf") else None


def value_on_board_fast(db, hand_arr, board_arr):
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
    board_key_map = {v: k for k, v in board_id_map.items()}
    lowest_value = float("inf")

    # Generate 2-card combos
    hand_combos = [
        "".join(sorted(combo, key=card_sort_key))
        for combo in combinations(hand_arr, 2)
    ]

    # Generate 3-card combos
    board_combos = [
        "".join(sorted(combo, key=card_sort_key))
        for combo in combinations(board_arr, 3)
    ]

    # Map board strings → ids (filtering out missing ones up front)
    board_ids = [board_id_map[b] for b in board_combos if b in board_id_map]

    for hand in hand_combos:
        hand_id = hand_id_map.get(hand)
        if hand_id is None:
            continue

        # Query DB once for all valid board_ids
        rows = db.get_evaluations(hand_id, board_ids)

        for board_id, _, value, rank in rows:
            board_key = board_key_map[board_id]  
            # board_key = next(k for k, v in board_id_map.items() if v == board_id)  # reverse lookup if needed
            # print(f"Board: {board_key}, Hand: {hand}, Hand Value: {value}, Rank: {rank}")
            lowest_value = min(lowest_value, value)

    return lowest_value if lowest_value != float("inf") else None

# def build_remaining_deck(hands, board=None):
#     deck = {r+s for r in "23456789TJQKA" for s in "shdc"}
#     used = set(card for hand in hands for card in hand)
#     if board:
#         used.update(board)
#     return list(deck - used)

def remaining_deck(all_cards, player_hands, board=None):
    used_cards = set(card for hand in player_hands for card in hand)
    if board:
        used_cards.update(board)
    return [c for c in all_cards if c not in used_cards]


def get_board_ids_for_boards(db, boards):
    board_id_map = db.get_board_ids()   # {board_str: board_id}
    return [board_id_map["".join(sorted(board, key=db.card_sort_key))] 
            for board in boards]

def generate_three_card_boards(deck, player_hands, board=None):
    """
    Generate all 3-card boards consistent with player hands and an optional board.
    
    Args:
        deck: List of all card strings in the deck
        player_hands: List of hands (each hand is a list of card strings)
        board: Optional list of cards (flop or turn)
    
    Returns:
        List of 3-card boards (each a tuple of 3 card strings)
    """
    # Remaining cards in deck
    rem_deck = remaining_deck(deck, player_hands, board)
    
    three_card_boards = set()
    
    if board is None:
        # No board given: all combinations of 3 from remaining deck
        three_card_boards.update(combinations(rem_deck, 3))
    else:
        board_len = len(board)
        if board_len == 3:
            # Flop
            three_card_boards.add(tuple(board))  # the flop itself
            # 2 flop + 1 from deck
            for two_flop in combinations(board, 2):
                for c in rem_deck:
                    three_card_boards.add(tuple(sorted(list(two_flop) + [c], key=card_sort_key)))
            # 1 flop + 2 from deck
            for one_flop in combinations(board, 1):
                for two_other in combinations(rem_deck, 2):
                    three_card_boards.add(tuple(sorted(list(one_flop) + list(two_other), key=card_sort_key)))
        elif board_len == 4:
            # Turn
            # Any 3-card subset of turn
            for three_from_turn in combinations(board, 3):
                three_card_boards.add(tuple(sorted(three_from_turn, key=card_sort_key)))
            # Any 2 from turn + 1 from remaining deck
            for two_from_turn in combinations(board, 2):
                for c in rem_deck:
                    three_card_boards.add(tuple(sorted(list(two_from_turn) + [c], key=card_sort_key)))
        else:
            raise ValueError("Board must be None, length 3 (flop), or length 4 (turn)")
    
    return list(three_card_boards)


def get_evaluations_for_hands_and_boards(db, hand_list, possible_boards):
    """
    Retrieve precomputed evaluations from the database for the given hands and boards.

    Args:
        db: Database connection / wrapper with cursor access.
        hand_list: List of hands (strings).
        possible_boards: List of board strings (3-card combos).

    Returns:
        dict mapping (hand, board) -> evaluation score/value
    """

    # First, convert hands and boards to their IDs
    hand_id_map = db.get_hand_ids()   # e.g. {'AsKsQsJh': 12345, ...}
    board_id_map = db.get_board_ids() # e.g. {'AsKsQs': 6789, ...}

    hand_ids = [hand_id_map[h] for h in hand_list if h in hand_id_map]
    board_ids = [board_id_map[b] for b in possible_boards if b in board_id_map]

    if not hand_ids or not board_ids:
        return {}

    # Build query for required evaluations
    # Assume `evaluations` table has schema: (hand_id, board_id, value)
    query = """
        SELECT hand_id, board_id, value
        FROM evaluations
        WHERE hand_id = ANY(%s) AND board_id = ANY(%s)
    """

    db.cursor.execute(query, (hand_ids, board_ids))
    rows = db.cursor.fetchall()

    # Build dictionary mapping back to string hands/boards
    reverse_hand_map = {v: k for k, v in hand_id_map.items()}
    reverse_board_map = {v: k for k, v in board_id_map.items()}

    eval_dict = {}
    for hand_id, board_id, value in rows:
        hand_str = reverse_hand_map[hand_id]
        board_str = reverse_board_map[board_id]
        eval_dict[(hand_str, board_str)] = value

    return eval_dict


def generate_possible_boards_with_weights(deck, board=None):
    """
    Generate all possible 3-card boards consistent with a given partial board.
    Returns list of (board_tuple, weight).
    """
    board = board or []
    board_set = set(board)
    remaining = [c for c in deck if c not in board_set]
    results = []

    if len(board) == 0:
        # Preflop: all 3-card combinations equally likely
        for combo in combinations(deck, 3):
            results.append((tuple(combo), 1.0))
        return results

    if len(board) == 3:
        # Flop known: must always include the exact flop (weight=1.0)
        results.append((tuple(board), 1.0))
        # Boards with 2 flop + 1 unknown
        for flop2 in combinations(board, 2):
            for other in remaining:
                results.append((tuple(sorted(flop2 + (other,))), 1.0 / len(remaining)))
        # Boards with 1 flop + 2 unknown
        for flop1 in board:
            for other2 in combinations(remaining, 2):
                results.append((tuple(sorted((flop1,) + other2)), 1.0 / comb(len(remaining), 2)))

    elif len(board) == 4:
        # Turn known: include all 3-card subsets of the 4 cards (weight=1.0 each)
        for combo in combinations(board, 3):
            results.append((tuple(combo), 1.0))
        # Boards with 2 turn cards + 1 unknown
        for flop2 in combinations(board, 2):
            for other in remaining:
                results.append((tuple(sorted(flop2 + (other,))), 1.0 / len(remaining)))

    else:
        raise ValueError("Only flop (3) or turn (4) supported.")

    return results


def calculate_equity(hands, deck, board=None, db_lookup=None):
    """
    Calculate equity for multiple hands given a possible partial board.
    - hands: list of hand tuples
    - deck: full deck of cards
    - board: list of known board cards (len 0, 3, or 4)
    - db_lookup: function(hand, board) -> (high_val, low_val)
    """
    board_weights = generate_possible_boards_with_weights(deck, board)
    scores = defaultdict(float)

    for board_combo, weight in board_weights:
        evals = []
        for hand in hands:
            high_val, low_val = db_lookup(hand, board_combo)
            evals.append((hand, high_val, low_val))

        # Find winners
        max_high = max(ev[1] for ev in evals)
        winners = [ev[0] for ev in evals if ev[1] == max_high]

        for w in winners:
            scores[w] += weight / len(winners)

    # Normalize equities
    total_weight = sum(weight for _, weight in board_weights)
    equities = {hand: score / total_weight for hand, score in scores.items()}

    return equities

def calculate_equity_for_multiple_hands(db, hand_list, board=None):
    # Step 1: Remaining deck
    remaining = build_remaining_deck(hand_list, board)
    
    # Step 2: Generate all possible boards
    possible_boards = generate_possible_boards(remaining, board)
    
    # Step 3: Map hands and boards to IDs
    hand_id_map = db.get_hand_ids()
    board_id_map = db.get_board_ids()
    
    hand_ids = [hand_id_map["".join(sorted(hand, key=db.card_sort_key))] 
                for hand in hand_list]
    board_ids = [board_id_map["".join(sorted(b, key=db.card_sort_key))] 
                 for b in possible_boards]

    # Step 4: Query DB for needed evaluations
    evals = db.get_evaluations_for_hands_and_boards(hand_ids, board_ids)
    
    # Step 5: Loop over boards → compare hand strengths
    equity_counts = [0] * len(hand_list)
    
    for b_id in board_ids:
        values = [evals[(h_id, b_id)] for h_id in hand_ids]
        max_val = max(values)
        winners = [i for i, v in enumerate(values) if v == max_val]
        for w in winners:
            equity_counts[w] += 1 / len(winners)  # split pots
    
    total = sum(equity_counts)
    equities = [c / total for c in equity_counts]
    return equities

def calculate_equity_for_multiple_hands(db, hand_list, board=None):
    """
    Calculates the equities for a list of hands.
    
    Arguments:
        db: The PLO evaluations database.
        hand_list: A list of lists of strings representing the hands being evaluated.
        board: Array of strings representing the cards for the board being evaluated.

    Returns:

        List/array of equities.
    """
    hand_id_map = db.get_hand_ids()
    board_id_map = db.get_board_ids()
    board_key_map = {v: k for k, v in board_id_map.items()}

    # Generate 2-card combos for each player

    # Generate potential board cards by removing player hand cards from deck

    # Generate 3-card combos from potential board cards

    # Retrieve evaluations for required hand combos for required boards

    # For each board:
    # value each players hand
    # determine winner(s) 
    
    # Determine equity for each hand

    # Return list / array of equities



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
    start_time = time.time()
    rank = value_on_board(db, ["As", "Kh", "Jc", "3c"], ["Ac", "Qd", "Jh", "Tc", "Qc"])
    end_time = time.time()
    print(f"Best hand: {rank}")
    print(f"Time taken: {end_time - start_time}")
 
    start_time = time.time()
    rank = value_on_board_fast(db, ["As", "Kh", "Jc", "3c"], ["Ac", "Qd", "Jh", "Tc", "Qc"])
    end_time = time.time()
    print(f"Best hand: {rank}")
    print(f"Time taken: {end_time - start_time}")

    # Close the DB connection
    db.close()


if __name__ == "__main__":
    main()
