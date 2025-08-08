RANK_ORDER = {r: i for i, r in enumerate('23456789TJQKA')}
SUIT_ORDER = {'s': 0, 'h': 1, 'd': 2, 'c':3}
SUITS = list(SUIT_ORDER.keys())
RANKS = list(RANK_ORDER.keys())

class Card:
    __slots__ = ['rank', 'suit']

    def __init__(self, rank, suit=None):
        if suit is None and isinstance(rank, str) and len(rank) == 2:
            self.rank = rank[0]
            self.suit = rank[1]
        else:
            self.rank = rank
            self.suit = suit

    # def rank_order(self):
    #     return {i for i, r in enumerate('23456789TJQKA')}
    
    def __repr__(self):
        return self.card_string()
    
    def card_string(self):
        return f"{self.rank}{self.suit}"

def card_sort_key(card):
    return (RANK_ORDER[card.rank], SUIT_ORDER[card.suit])
