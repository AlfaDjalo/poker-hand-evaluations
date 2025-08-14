from db import DB, open_db
# import time
# import matplotlib.pyplot as plt
# import numpy as np
from collections import defaultdict
from itertools import combinations, product, permutations, count
from phevaluator import evaluate_cards

from card import card_sort_key, SUITS
# from card import Card, card_sort_key, RANK_ORDER, SUIT_ORDER, SUITS, RANKS
from deck import Deck

# BOARD_PATTERNS = [[5, 0, 0, 0], [4, 1, 0, 0]]
BOARD_PATTERNS = [[5, 0, 0, 0], [4, 1, 0, 0], [3, 2, 0, 0], [3, 1, 1, 0], [2, 2, 1, 0], [2, 1, 1, 1]]
PATTERN_COUNTS = [4, 12, 12, 12, 12, 14]

def create_hands_table(db):
    """
    Clear the hands table and repopulate.
    """
    db.truncate_table("hands")  
    hands = generate_hands()
    print(hands[:10])
    hand_id_map = db.bulk_insert_hands(hands)
    return

def create_boards_table(db):
    """
    Clear the boards table and repopulate.
    """
    # db.drop_table("boards")
    # db.init_schema()
    db.truncate_table("boards")
    for pattern_num, pattern in enumerate(BOARD_PATTERNS):
        boards = generate_boards_by_suit_distribution(pattern)
        board_id_map = db.bulk_insert_boards(boards, pattern_num)
    return

def create_evaluations_table(db):
    """
    Clear the evaluations table and repopulate.
    """
    hand_id_map = db.get_hand_ids()
    db.truncate_table("evaluations")
    try:
        for i, pattern in enumerate(BOARD_PATTERNS):
            print(f"Running board pattern {i}")
            board_id_map = db.get_board_ids(i)
            print(len(board_id_map))
            all_hand_values = run_evaluations(db, hand_id_map, board_id_map)
    except Exception as e:
            import traceback
            print(f"An error occurred during evaluation: {e}")
            traceback.print_exc()
            # You might want to log the error or perform a rollback here

    # db.replace_indices_on_evaluations()

    return

def generate_boards_by_suit_distribution(suit_counts):
    """
    Generates canonical list of boards based on number of cards of each suit.
    Returns a list of board strings (e.g., "AsKsQsJhTh").
    """
    my_deck = Deck()
    deck = my_deck.get_cards()

    assert sum(suit_counts) == 5
    assert len(suit_counts) == 4

    suit_to_cards = {suit: [card for card in deck if card.suit == suit] for suit in SUITS}

    suit_combos = [list(combinations(suit_to_cards[suit], count)) for suit, count in zip(SUITS, suit_counts) if count > 0]

    raw_boards = product(*suit_combos)
    # Return as string representation
    boards = ["".join(card.card_string() for group in combo for card in group) for combo in raw_boards]

    return boards

def generate_hands():
    """
    Returns all possible hand strings (e.g., "8s8c").
    """

    my_deck = Deck()
    deck_cards = my_deck.get_cards()

    hands = []
    for c1, c2 in combinations(deck_cards, 2):
        cards = sorted((c1, c2), key=card_sort_key)
        hands.append(cards[0].card_string() + cards[1].card_string())

    return hands

def run_evaluations(db, hand_id_map, board_id_map):

    all_evaluations_to_insert = []

    for board_str, board_id in board_id_map.items():
        hand_values = evaluate_board(board_str, hand_id_map)
        hand_rankings = rank_hands_for_board(hand_values)

        evaluations_for_board = [
            (board_id, hand_id, hand_value, min_rank, max_rank, avg_rank, dense_rank)
            for hand_id, hand_value, min_rank, max_rank, avg_rank, dense_rank in hand_rankings
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
    board_cards = [board_str[i:i+2] for i in range(0, 10, 2)]
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
    num_total_hands = len(hand_values)

    denom_non_dense = num_total_hands - 1 if num_total_hands > 1 else 1
    denom_dense = num_unique_values - 1 if num_unique_values > 1 else 1

    pos = 0
    for rank_index, value in enumerate(sorted_values):
        indices = value_to_indices[value]

        min_rank_val = pos
        max_rank_val = pos + len(indices) - 1
        avg_rank_val = pos + (len(indices) - 1) / 2
        dense_rank_val = rank_index

        min_percentile = min_rank_val / denom_non_dense
        max_percentile = max_rank_val / denom_non_dense
        avg_percentile = avg_rank_val / denom_non_dense
        dense_percentile = dense_rank_val / denom_dense

        for original_index in indices:
            h, v = hand_values[original_index]
            hand_rankings[original_index] = (h, v, min_percentile, max_percentile, avg_percentile, dense_percentile)

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

    # create_hands_table(db)
    # create_boards_table(db)
    create_evaluations_table(db)
    # check_evaluations_for_hand(db, "AhKd")
    # check_evaluations_for_hand(db, "AhKd")
    # plot_chart_for_hand(db, "7h2c", "rank_min")
    # plot_rank_distribution(db, "7h7c")
    # plot_rank_distribution_multi2(db, ["AQo", "JTs", "99"])
    # Close the DB connection
    db.close()


if __name__ == "__main__":
    main()
