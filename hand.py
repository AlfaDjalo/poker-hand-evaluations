from card import Card, card_sort_key

class Hand:
    __slots__ = ['card1', 'card2']

    def __init__(self, card1, card2):
        self.card1, self.card2 = sorted((card1, card2), key=card_sort_key)

    # def _card_sort_key(self, card):
    #     return (Card.rank_order[card.rank], card.suit)

    def __iter__(self):
        return iter((self.card1, self.card2))
    
    def __repr__(self):
        return f"{self.card1}{self.card2}"
    
    def to_str(self):
        return f"{self.card1}{self.card2}"

    # def __lt__(self, other):
    #     self_sorted = sorted((self.card1, self.card2), key = self._card_sort_key)
    #     other_sorted = sorted((other.card1, other.card2), key = self._card_sort_key)
    #     return (self_sorted < other_sorted)

    # @property   
    # def cards(self):
    #     return (self.card1, self.card2)