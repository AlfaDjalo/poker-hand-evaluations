import time
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from itertools import combinations, product, permutations, count
from phevaluator import evaluate_cards
from collections import defaultdict

from card import Card, card_sort_key, RANK_ORDER, SUIT_ORDER, SUITS, RANKS
from deck import Deck
from hand import Hand
from board import Board
from db import DB

# SUITS = ['s', 'h', 'd', 'c']
# RANKS = []
BOARD_PATTERNS = [[5, 0, 0, 0]]
# BOARD_PATTERNS = [[5, 0, 0, 0], [4, 1, 0, 0], [3, 2, 0, 0], [3, 1, 1, 0], [2, 1, 1, 1]]


def generate_boards_by_suit_distribution(deck, suit_counts):
    """
    Generates canonical list of boards based on number of cards of each suit.
    Returns a list of board strings (e.g., "AsKsQsJhTh").
    """

    assert sum(suit_counts) == 5
    assert len(suit_counts) == 4

    suit_to_cards = {suit: [card for card in deck if card.suit == suit] for suit in SUITS}

    suit_combos = [list(combinations(suit_to_cards[suit], count)) for suit, count in zip(SUITS, suit_counts) if count > 0]

    raw_boards = product(*suit_combos)
    # Return as string representation
    boards = ["".join(card.card_string() for group in combo for card in group) for combo in raw_boards]

    return boards


def generate_hands_by_suit_distribution(suit_counts):
    """
    Generates canonical list of hands for the given board set.
    Hands in the same hand category would need to use the same approach
    for replication to other suit combinations.
    
    Arguments:
        deck: Deck object.
        suit_counts: list containing the number of cards for each suit on the board,
        starting from the most frequent suit downwards.
        
    Returns:
        A dictionary with a list of hands for each category. 
    """

    assert sum(suit_counts) == 5
    assert len(suit_counts) == 4

    suits_in_board = [suit for suit, count in zip(SUITS, suit_counts) if count > 0]

    main_suit = suits_in_board[0]
    other_suits = suits_in_board[1:]
    non_board_suits = [suit for suit in SUITS if suit not in suits_in_board]

    rank_combos = [(rank1, rank2) for i, rank1 in enumerate(RANKS) for j, rank2 in enumerate(RANKS) if i <= j]



    hand_combos = rank_combos

    return hand_combos

def get_hands():
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

# Old function, uses old hand list.
# def evaluate_board(board, hands):
#     """
#     Evaluate all hands on a single board.

#     Arguments:
#         board: a tuple of five cards representing the board to be evaluated.
#         hands: a list of tuples of two cards representing the hand to be evaluated.

#     Returns:
#         A list of tuples of (hand, hand value)
#     """
#     board_set = set(board)
#     hand_values = []
#     for hand in hands:
#         if hand.card1 in board_set or hand.card2 in board_set:
#             hand_values.append((hand.to_str(), float('inf')))
#         else:
#             full_hand = tuple(hand) + tuple(board)
#             # full_hand = hand.cards + tuple(board)
#             # print(full_hand)
#             hand_values.append((hand.to_str(), evaluate_hand(full_hand)))
#             # hand_values.append(evaluate(*full_hand))
#     return hand_values

def evaluate_board(board_str, hand_id_map):
    """
    Evaluate all hands on a single board, using hand_id_map only.

    Arguments:
        board_str: string of 10 chars, e.g. "AsKsQsJhTh"
        hand_id_map: dict mapping hand_str (e.g. "8s8c") to hand_id

    Returns:
        List of tuples: (hand_id, hand_value)
    """
    board_cards = [Card(board_str[i:i+2]) for i in range(0, 10, 2)]
    board_set = set(board_cards)
    hand_values = []

    for hand_str, hand_id in hand_id_map.items():
        card1 = Card(hand_str[:2])
        card2 = Card(hand_str[2:])

        if card1 in board_set or card2 in board_set:
            hand_values.append((hand_id, float('inf')))
        else:
            full_hand = (card1, card2) + tuple(board_cards)
            hand_value = evaluate_hand(full_hand)
            hand_values.append((hand_id, hand_value))

    return hand_values


def evaluate_hand(cards):
    """
    Evaluates the best holdem style hand on a seven card board + hand.
    
    Arguments:
        cards: a seven card tuple of cards representing the board + hand to be evaluated.

    Returns:
        the value of the best hand.
    """

    evaluate = evaluate_cards
    full_hand = [card.card_string() for card in cards]
    return evaluate(*full_hand)


def rank_hands_on_all_boards(hand_values, method="min", return_percentiles=True):
    """
    Iterates through all board, ranking all hands on each board.

    Arguments:
        hand values:
        method:
        return_percentiles:

    Returns:
        a list of ????
    """

    ranked_hand_values = []

    for board_hand_values in hand_values:
        ranked_hand_values.append(rank_hands_for_board(board_hand_values, method=method, return_percentiles=return_percentiles))

    return ranked_hand_values


def rank_hands_for_board(hand_values, method="min", return_percentiles=True):

    hand_values.sort(key=lambda x: x[1])
    valid_hands = [(i, hand, value) for i, (hand, value) in enumerate(hand_values) if value != float('inf')]
    values = [value for _, _, value in valid_hands]

    value_to_indices = defaultdict(list)
    for idx, (i, hand, val) in enumerate(valid_hands):
        value_to_indices[val].append(i)

    sorted_values = sorted(value_to_indices.keys())
    hand_rankings = [None] * len(hand_values)

    if method == "dense":
        num_for_percentiles = len(sorted_values)
    else:
        num_for_percentiles = len(valid_hands)

    pos = 0
    for rank_index, value in enumerate(sorted_values):
        indices = value_to_indices[value]
        match method:
            case "min":
                rank = pos
            case "max":
                rank = pos + len(indices) - 1
            case "average":
                rank = pos + (len(indices) - 1) / 2
            case "dense":
                rank = rank_index
            case _:
                raise ValueError("Unknown ranking method.")

        for i in indices:
            h, v = hand_values[i]
            p = rank / (num_for_percentiles - 1) if return_percentiles else rank
            hand_rankings[i] = (h, v, p)

        pos += len(indices)

    for i, (h, v) in enumerate(hand_values):
        if v == float('inf'):
            hand_rankings[i] = (h, v, None)

    return hand_rankings


def hand_distribution(hand_str, hand_rankings_on_board_set):
    """
    hand_str: e.g. "8s8c"
    hand_rankings_on_board_set: list of lists of (hand_id, value, rank)
    Returns: list of (value, rank) for the given hand_str
    """


    distribution = []

    for board in hand_rankings_on_board_set:
        for h, v, r in board:
            if h == hand_str and r is not None:
                distribution.append((v, r))

    return distribution


def plot_rank_distribution(distribution, bins=20, title=None):
    """
    Plots a histogram of the rank of a single hand on a set of boards.
    
    Arguments:
        distribution: a list containing the distribution of hand rankings.
        bins: the number of bins in the histogram.
        title: the title to be printed on the chart.
    """



    ranks = [rank * 100 for value, rank in distribution if rank is not None]
    # ranks = [rank for _, result in distribution if result is not None and result[1] is not None for rank in [result[1]]]

    if not ranks:
        print("No valid ranks to plot.")
        return
    
    plt.figure(figsize=(8, 4))
    plt.hist(ranks, bins=bins, edgecolor='black', range=(0, 100))
    plt.xlabel("Percentile Rank")
    plt.ylabel("Frequency")
    plt.title(title or "Hand Percentile Rank Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return

def run_evaluation(hand_id_map):

    start_time = time.time()

    my_deck = Deck()
    deck_cards = my_deck.get_cards()

    # hands = [Hand(*sorted((c1, c2), key = card_sort_key)) for c1, c2 in combinations(deck_cards, 2)]

    hands = hand_id_map
    # hands = get_hands()

    for pattern_num, pattern in enumerate(BOARD_PATTERNS):
        boards = generate_boards_by_suit_distribution(deck_cards, pattern)
        print(f"{len(boards)} boards match the pattern")
        hand_values = [None] * len(boards)

        for board_num, board_str in enumerate(boards):
            hand_values[board_num] = evaluate_board(board_str, hands)

        print(len(hand_values[board_num]))
        print(len(hand_values))

    end_time = time.time()

    time_taken = end_time - start_time
    print(f"Time taken: {time_taken}")
    print(f"Average time per board: {time_taken / len(hand_values)}")

    return hand_values



def run_distribution(hand_values, method='min', return_percentiles=True):


    start_time = time.time()

    hand_rankings_on_all_boards = rank_hands_on_all_boards(hand_values, method=method, return_percentiles=return_percentiles)

    end_time = time.time()

    time_taken = end_time - start_time
    print(f"Time taken: {time_taken}")

    distribution = hand_distribution("8s8c", hand_rankings_on_all_boards)
    # print(distribution[:10])

    return distribution


def test_database():
    # Initialize DB connection
    db = DB(
        dbname="postgres",
        user="postgres",
        password="**********",  # Be sure to protect this in production
        host="localhost",
        port="5432"
    )

    try:
        # Initialize tables
        db.init_schema()

        # Insert a test board and hand
        test_board = "AsKsQsJhTh"
        test_hand = "9c9d"

        db.insert_board(test_board)
        db.insert_hand(test_hand)

        # You can manually check the IDs in pgAdmin or write SELECT queries
        # For demonstration, let's assume board_id = 1 and hand_id = 1
        db.insert_evaluation(
            board_id=1,
            hand_id=1,
            hand_value=1234,
            rank_min=0.01,
            rank_max=0.05,
            rank_avg=0.03,
            rank_dense=0.02
        )

        # Run a test query
        db.select_query()

    finally:
        # Clean up connection
        db.close()


def test_bulk_load_hands():
    # Initialize DB connection
    db = DB(
        dbname="postgres",
        user="postgres",
        password="**********",  # Be sure to protect this in production
        host="localhost",
        port="5432"
    )

    my_deck = Deck()
    deck_cards = my_deck.get_cards()
    hands = [Hand(*sorted((c1, c2), key = card_sort_key)) for c1, c2 in combinations(deck_cards, 2)]
    data = [(hand.to_str(),) for hand in hands]  # or hand.to_str() if defined
    print(len(data))
    db.select_hands()
    db.truncate_table("hands")
    db.select_hands()
    hand_id_map = db.bulk_insert_hands(data)
    db.select_hands()
    # print(hand_id_map)

    db.close()

    return hand_id_map


def test_bulk_load_boards():
    # Initialize DB connection
    db = DB(
        dbname="postgres",
        user="postgres",
        password="***********",  # Be sure to protect this in production
        host="localhost",
        port="5432"
    )

    my_deck = Deck()
    deck_cards = my_deck.get_cards()

    for pattern_num, pattern in enumerate(BOARD_PATTERNS):
        boards = generate_boards_by_suit_distribution(deck_cards, pattern)
    print(boards[0])
    # data = [(board.to_str(),) for board in boards]  # or hand.to_str() if defined
    print(len(boards))
    db.select_boards()
    db.truncate_table("boards")
    db.select_boards()
    board_id_map = db.bulk_insert_boards(boards, pattern_num)
    db.select_boards()
    # print(hand_id_map)

    db.close()

    return board_id_map


def load_db_config(config_path="db_config.json"):
    """
    Loads DB config from a JSON file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Database config file '{config_path}' not found.")
    with open(config_path, "r") as f:
        return json.load(f)

def open_db():
    """
    Opens database object.

    Returns:
        Database
    """
    config = load_db_config()
    db = DB(
        dbname=config["dbname"],
        user=config["user"],
        password=config["password"],
        host=config.get("host", "localhost"),
        port=config.get("port", 5432)
    )
    return db

def main():

    config = load_db_config()
    db = DB(
        dbname=config["dbname"],
        user=config["user"],
        password=config["password"],
        host=config.get("host", "localhost"),
        port=config.get("port", 5432)
    )

    # test_database()
    # hand_id_map = test_bulk_load_hands()
    # hand_id_map = db.get_hand_ids()

    board_id_map = test_bulk_load_boards()
    board_id_map = db.get_board_ids()
    
    # hand_combos = generate_hands_by_suit_distribution(BOARD_PATTERNS[0])

    # print(hand_combos)
    # hand_values = run_evaluation(hand_id_map)

    # distribution = run_distribution(hand_values, method='min', return_percentiles=True)

    # plot_rank_distribution(distribution, bins=20, title="8s8c Hand Strength Across Boards - Best Ranking")

    return

if __name__ == "__main__":
    main()