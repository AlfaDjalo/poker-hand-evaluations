from db_plo import DB_PLO, open_db

from treys import Evaluator, Card
from card import SUITS, RANKS
from itertools import combinations, islice

from low_hand_evaluator import LowEvaluator
import time
# import matplotlib.pyplot as plt
# import numpy as np
# from collections import defaultdict, Counter
# from itertools import combinations, product, permutations, count
# from phevaluator import evaluate_cards

# from card import card_sort_key, SUITS
# # from card import Card, card_sort_key, RANK_ORDER, SUIT_ORDER, SUITS, RANKS
# from deck import Deck

# BOARD_PATTERNS = [[3, 0, 0], [2, 1, 0], [1, 1, 1]]
# PATTERN_COUNTS = [4, 12, 12, 12, 12, 14]

def create_evaluations_table(db):
    """
    Clear the evaluations table and repopulate.
    """
    # Build both string deck and Treys deck
    deck = [r + s for r in RANKS for s in SUITS]
    treys_deck = [Card.new(c) for c in deck]

    # ---- Hands ----
    hand_map = {}
    for hand_strs, hand_treys in zip(combinations(deck, 2),
                                     combinations(treys_deck, 2)):
        hand_id = cards_to_bitmask(hand_strs)           # DB key
        hand_map[hand_id] = hand_treys                  # Treys tuple

    print(f"Number of hands: {len(hand_map)}")

    # ---- Boards ----
    board_map = {}
    for board_strs, board_treys in zip(combinations(deck, 3),
                                       combinations(treys_deck, 3)):
        board_id = cards_to_bitmask(board_strs)         # DB key
        board_map[board_id] = board_treys               # Treys tuple

    print(f"Number of boards: {len(board_map)}")

    db.truncate_table("plo_evaluations_bm")

    run_evaluations(db, hand_map, board_map)

    return

def run_evaluations(db, hand_map, board_map, batch_size=500000000):

    high_evaluator = Evaluator()
    low_evaluator = LowEvaluator()
    batch = []

    # all_evaluations_to_insert = []

    for board_id, board_treys in board_map.items():
        for hand_id, hand_treys in hand_map.items():
            # Skip if hand and board overlap
            if hand_id & board_id:
                continue

            high_hand_value = high_evaluator.evaluate(list(hand_treys), list(board_treys))
            low_hand_value = low_evaluator.evaluate(hand_treys + board_treys)
            batch.append((board_id, hand_id, high_hand_value, low_hand_value))

            if len(batch) >= batch_size:
                db.bulk_insert_evaluations(batch)
                batch.clear()

    if batch:
        db.bulk_insert_evaluations(batch)

    return

def evaluate_board(evaluator, board_id, hand_ids):
    """
    Evaluate all hands on a single board, using hand_id_map only.

    Arguments:
        board_str: string of 10 chars, e.g. "AsKsQsJhTh"
        hand_id_map: dict mapping hand_str (e.g. "8s8c") to hand_id

    Returns:
        List of tuples: (hand_id, hand_value)
    """
    # board_cards = [board_str[i:i+2] for i in range(0, 6, 2)]
    # board_set = set(board_cards)
    hand_values = []
    for hand_id in hand_ids:
        # print(hand_id)
        # print(board_id)
        if hand_board_no_overlap(hand_id, board_id):
            hand_value = evaluate_hand(evaluator, board_id, hand_id)
            hand_values.append((hand_id, hand_value))
    return hand_values


def evaluate_hand(evaluator, board_id, hand_id):
    """
    Evaluates the best omaha style hand on a three card board + hand.
    Arguments:
        cards: a list of 5 card strings (e.g. ["As", "Ks", ...])
    Returns:
        the value of the best hand.
    """
    print(board_id)
    print(hand_id)
    return evaluator.evaluate(board_id, hand_id)


def cards_to_bitmask(cards):
    """
    Convert a list of card strings to a bitmask representation.
    
    Args:
        cards: List of card strings like ['As', 'Kh', 'Qd']
    
    Returns:
        int: Bitmask where each card is represented by a unique bit
        
    Card encoding: 4 bits per rank (one per suit)
    Ranks: A=0-3, K=4-7, Q=8-11, J=12-15, T=16-19, 9=20-23, ..., 2=48-51
    Within each rank: spades=+0, hearts=+1, diamonds=+2, clubs=+3
    """
    RANK_OFFSETS = {
        'A': 0, 'K': 4, 'Q': 8, 'J': 12, 'T': 16, 
        '9': 20, '8': 24, '7': 28, '6': 32, '5': 36, 
        '4': 40, '3': 44, '2': 48
    }
    
    SUIT_OFFSETS = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
    
    bitmask = 0
    for card in cards:
        if len(card) != 2:
            raise ValueError(f"Invalid card format: {card}")
        
        rank, suit = card[0], card[1]
        if rank not in RANK_OFFSETS or suit not in SUIT_OFFSETS:
            raise ValueError(f"Invalid card: {card}")
        
        bit_position = RANK_OFFSETS[rank] + SUIT_OFFSETS[suit]
        bitmask |= (1 << bit_position)
    
    return bitmask


def bitmask_to_cards(bitmask):
    """
    Convert a bitmask back to a list of card strings (for debugging/verification).
    
    Args:
        bitmask: int bitmask representation
    
    Returns:
        List of card strings
    """   
    cards = []
    for rank_idx, rank in enumerate(RANKS):
        for suit_idx, suit in enumerate(SUITS):
            bit_position = rank_idx * 4 + suit_idx
            if bitmask & (1 << bit_position):
                cards.append(rank + suit)
    
    return cards

def hand_board_no_overlap(hand_id, board_id):
    """
    Check if any cards are present in both hand and board.

    Args:
        hand_id: List of card strings like ['As', 'Kh', 'Qd', 'Jc']
        board_id: List of card strings like ['Tc', '7d', '3h']

    Returns:
        bool: True if any card is in both hand and board, False otherwise
    """
    return (hand_id & board_id) != 0

def hand_stream(card_bitmasks):
    for h1, h2 in combinations(card_bitmasks, 2):
        yield h1 | h2  # OR bitmasks to get a single integer ID

def board_stream(card_bitmasks):
    for b1, b2, b3 in combinations(card_bitmasks, 3):
        yield b1 | b2 | b3  # single integer ID


def main():

    start_time = time.time()

    # Initialize DB connection
    db = open_db()

    # db.add_indices_on_evaluations()
    # db.drop_table("plo_boards")
    # db.drop_table("plo_evaluations")
    # db.init_schema()
    # create_boards_table(db)
    create_evaluations_table(db)
    # Close the DB connection
    db.close()

    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")

if __name__ == "__main__":
    main()
