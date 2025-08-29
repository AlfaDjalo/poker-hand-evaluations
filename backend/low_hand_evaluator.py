from itertools import combinations
from treys import Card

class LowEvaluator:
    """
    Evaluates Ace-to-Five low hands (Omaha8 / Stud8 style).
    """

    def __init__(self, eight_or_better=True):
        self.eight_or_better = eight_or_better

    def _ranks_from_cards(self, cards):
        """Extract ranks 0..12 (Ace=12, Two=0)."""
        return [Card.get_rank_int(c) for c in cards]

    @staticmethod
    def _treys_to_low_rank(card):
        """Convert Treys card int to Ace=1..8 for low evaluation, otherwise None."""
        rank = Card.get_rank_int(card)  # 0=2 ... 12=Ace
        ace_low = 1
        r = rank + 2 if rank != 12 else ace_low  # 2..14, Ace=1
        if r <= 8:
            return r
        return None

    def evaluate(self, cards):
        """
        Return tuple of sorted low cards for Ace-to-Five low, or None if no qualifying low.
        """
        low_candidates = [_ for _ in (self._treys_to_low_rank(c) for c in cards) if _ is not None]
        best = None

        for combo in combinations(low_candidates, 5):
            if len(set(combo)) < 5:
                continue  # must have 5 distinct ranks
            combo_sorted = tuple(sorted(combo, reverse=True))
            if best is None or combo_sorted < best:
                best = combo_sorted

        if best is None:
            return 99999
        
        return int("".join(str(r) for r in best))  # tuple of 5 ints, descending, lower is better


    # def evaluate(self, cards):
    #     """
    #     Given a list of 5-7 Treys int-cards, return a low-hand score.
    #     Lower score is better.  None if no qualifying low.
    #     """
    #     ranks = self._ranks_from_cards(cards)

    #     # Convert Ace-high ranks (0..12, Ace=12) into Ace=1 for lowball
    #     low_ranks = [(1 if r == 12 else r + 2) for r in ranks]  # 2..14 → 2..14, Ace=1
    #     low_ranks = [r if r <= 8 else None for r in low_ranks]  # filter >8

    #     if not any(low_ranks):
    #         return None

    #     # Pick best 5-card low
    #     best = None
    #     for combo in combinations([c for c in low_ranks if c is not None], 5):
    #         if len(set(combo)) < 5:
    #             continue  # must be 5 unique ranks
    #         # Normalize into descending order
    #         hand_tuple = tuple(sorted(combo, reverse=True))
    #         if best is None or hand_tuple < best:
    #             best = hand_tuple

    #     return best  # e.g. (8, 6, 4, 2, 1)

    # def hand_rank(self, cards):
    #     """
    #     Return integer rank (smaller = stronger).
    #     """
    #     score = self.evaluate(cards)
    #     if score is None:
    #         return None
    #     # Convert tuple into comparable integer
    #     return sum(v * (15 ** i) for i, v in enumerate(score))


# Example usage
if __name__ == "__main__":
    le = LowEvaluator()

    board = [Card.new("2h"), Card.new("7d"), Card.new("6c"), Card.new("Ks"), Card.new("5h")]
    hand = [Card.new("Ah"), Card.new("3c"), Card.new("Qc"), Card.new("5d")]

    # Evaluate Omaha low
    # In Omaha you must pick 2 from hand + 3 from board
    best = None
    for hand_combo in combinations(hand, 2):
        for board_combo in combinations(board, 3):
            best = le.evaluate_low(hand_combo + board_combo)
            print("Best low score:", best)
            
            # val = le.hand_rank(hand_combo + board_combo)
            # if val is not None and (best is None or val < best):
            #     best = val

