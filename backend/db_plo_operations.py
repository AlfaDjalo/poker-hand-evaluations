from db_plo import DB_PLO, open_db
# import time
# import matplotlib.pyplot as plt
# import numpy as np
from collections import defaultdict, Counter
from itertools import combinations, product, permutations, count
from phevaluator import evaluate_cards

from card import card_sort_key, SUITS
# from card import Card, card_sort_key, RANK_ORDER, SUIT_ORDER, SUITS, RANKS
from deck import Deck

BOARD_PATTERNS = [[3, 0, 0], [2, 1, 0], [1, 1, 1]]
PATTERN_COUNTS = [4, 12, 12, 12, 12, 14]

# def create_hands_table(db):
#     """
#     Clear the hands table and repopulate.
#     """
#     db.truncate_table("hands")  
#     hands = generate_hands()
#     print(hands[:10])
#     hand_id_map = db.bulk_insert_hands(hands)
#     return

def create_evaluations_table(db):
    """
    Clear the evaluations table and repopulate.
    """
    hand_id_map = db.get_hand_ids()
    db.truncate_table("plo_evaluations")
    try:
        for i, pattern in enumerate(BOARD_PATTERNS):
            print(f"Running board pattern {i+1}")
            board_id_map = db.get_board_ids(i+1)
            print(len(board_id_map))
            all_hand_values = run_evaluations(db, hand_id_map, board_id_map)
    except Exception as e:
            import traceback
            print(f"An error occurred during evaluation: {e}")
            traceback.print_exc()
            # You might want to log the error or perform a rollback here

    # db.replace_indices_on_evaluations()

    return

def create_boards_table(db):
    """
    Generate all possible 3-card flop combinations and insert them into the database.
    Pattern number is based on most common suit count:
    - 3: monotone (all same suit)
    - 2: flush draw (two of same suit)
    - 1: rainbow (all different suits)
    """
    # Assuming you have a Deck class and SUITS constant
    my_deck = Deck()
    deck = my_deck.get_cards()
    
    # Clear the boards table
    db.truncate_table("plo_boards")
    
    # Generate all possible 3-card combinations
    all_flops = list(combinations(deck, 3))
    
    # Prepare data for bulk insert
    flop_data = []

    for flop in all_flops:
        # Count suits in this flop
        suits = [card.suit for card in flop]
        suit_counts = Counter(suits)
        
        # Get the most common suit count
        max_suit_count = max(suit_counts.values())
        
        # Sort cards for consistent ordering (optional but recommended)
        sorted_flop = sorted(flop, key=card_sort_key)

        flop_data.append({
            'card1_str': sorted_flop[0].card_string(),
            'card2_str': sorted_flop[1].card_string(),
            'card3_str': sorted_flop[2].card_string(),
            'suit_pattern': max_suit_count
        })
    
    # Bulk insert all flops
    board_ids = db.bulk_insert_flops(flop_data)
    print(f"Inserted {len(flop_data)} total flops")
    # print(board_ids)
    return

def run_evaluations(db, hand_id_map, board_id_map):

    all_evaluations_to_insert = []

    for board_str, board_id in board_id_map.items():
        hand_values = evaluate_board(board_str, hand_id_map)
        hand_rankings = rank_hands_for_board(hand_values)

        evaluations_for_board = [
            (board_id, hand_id, hand_value, dense_rank)
            for hand_id, hand_value, dense_rank in hand_rankings
        ]

        all_evaluations_to_insert.extend(evaluations_for_board)

    db.bulk_insert_evaluations(all_evaluations_to_insert)

    return None

def evaluate_board(board_str, hand_id_map):
    """
    Evaluate all hands on a single board, using hand_id_map only.

    Arguments:
        board_str: string of 10 chars, e.g. "AsKsQsJhTh"
        hand_id_map: dict mapping hand_str (e.g. "8s8c") to hand_id

    Returns:
        List of tuples: (hand_id, hand_value)
    """
    board_cards = [board_str[i:i+2] for i in range(0, 6, 2)]
    board_set = set(board_cards)
    hand_values = []
    for hand_str, hand_id in hand_id_map.items():
        card1 = hand_str[:2]
        card2 = hand_str[2:]
        if card1 not in board_set and card2 not in board_set:
            full_hand = [card1, card2] + board_cards
            hand_value = evaluate_hand(full_hand)
            hand_values.append((hand_id, hand_value))
    return hand_values

def rank_hands_for_board(hand_values):
    """
    Function to calculated rankings for each hand on a given board.
    
    Args:
        hand_values: List of tuples of (hand _id, value)

    Returns:
        List of tuples of (hand_id, value, rank_min, rank_max, rank_avg, rank_dense)
    """
    hand_values.sort(key=lambda x: x[1])
    value_to_indices = defaultdict(list)
    for idx, (hand_id, value) in enumerate(hand_values):
        value_to_indices[value].append(idx)

    sorted_values = sorted(value_to_indices.keys())
    hand_rankings = [None] * len(hand_values)

    num_unique_values = len(sorted_values)

    denom_dense = num_unique_values - 1 if num_unique_values > 1 else 1

    pos = 0
    for rank_index, value in enumerate(sorted_values):
        indices = value_to_indices[value]

        dense_rank_val = rank_index

        dense_percentile = dense_rank_val / denom_dense

        for original_index in indices:
            h, v = hand_values[original_index]
            hand_rankings[original_index] = (h, v, dense_percentile)

        pos += len(indices)

    return hand_rankings

def evaluate_hand(cards):
    """
    Evaluates the best holdem style hand on a seven card board + hand.
    Arguments:
        cards: a list of 7 card strings (e.g. ["As", "Ks", ...])
    Returns:
        the value of the best hand.
    """

    evaluate = evaluate_cards
    # cards is already a list of strings
    return evaluate(*cards)


def main():

    # Initialize DB connection
    db = open_db()

    # db.drop_table("plo_boards")
    # db.drop_table("plo_evaluations")
    # db.init_schema()
    create_boards_table(db)
    create_evaluations_table(db)
    # Close the DB connection
    db.close()


if __name__ == "__main__":
    main()
