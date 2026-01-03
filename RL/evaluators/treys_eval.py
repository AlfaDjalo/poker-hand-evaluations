from itertools import combinations
from treys import Evaluator, Card

def evaluate_hand_old(board_cards, hand_cards):
    """
    Evaluates the best omaha style hand on a three card board + hand.
    Arguments:
        cards: a list of 5 card strings (e.g. ["As", "Ks", ...])
    Returns:
        the value of the best hand.
    """
    evaluator = Evaluator()

    hand = [Card.new(hand_card) for hand_card in hand_cards]
    board = [Card.new(board_card) for board_card in board_cards]



    return evaluator.evaluate(board, hand)



def evaluate_hand(hand_cards, board_cards, showdown_style="holdem"):
    """
    Evaluate the best possible 5-card poker hand.

    Parameters
    ----------
    hand_cards : list[str]
        Hole cards as strings, e.g. ["Ah", "Qs"]
    board_cards : list[str]
        Board cards as strings, e.g. ["As", "Kd", "7h", ...]
    showdown_style : str
        - "holdem" (default): best 5-card hand using any cards
        - "omaha": exactly 2 hole cards + exactly 3 board cards

    Returns
    -------
    int
        Treys hand rank (lower is better)
    """

    evaluator = Evaluator()

    # Convert once
    hand = [Card.new(c) for c in hand_cards]
    board = [Card.new(c) for c in board_cards]

    if showdown_style == "omaha":
        if len(hand) < 2:
            raise ValueError("Omaha requires at least 2 hole cards")
        if len(board) < 3:
            raise ValueError("Omaha requires at least 3 board cards")

        best_rank = None

        # Exactly 2 from hand, exactly 3 from board
        for hand_combo in combinations(hand, 2):
            for board_combo in combinations(board, 3):
                rank = evaluator.evaluate(
                    list(hand_combo),
                    list(board_combo),
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank

        return best_rank

    elif showdown_style == "holdem":
        all_cards = hand + board

        if len(all_cards) < 5:
            raise ValueError("Hold'em evaluation requires at least 5 total cards")

        best_rank = None

        # Any 5-card combination
        for five_cards in combinations(all_cards, 5):
            # Treys wants board + hand, but this is just a partition;
            # we can pass all 5 as 'board' and empty hand.
            rank = evaluator.evaluate(list(five_cards), [])
            if best_rank is None or rank < best_rank:
                best_rank = rank

        return best_rank

    else:
        raise ValueError(f"Unknown showdown_style: {showdown_style}")
