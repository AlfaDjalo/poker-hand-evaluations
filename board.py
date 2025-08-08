
class Board:
    __slots__ = ['cards']

    def __init__(self, cards):
        self.cards = cards

    def __repr__(self):
        return ''.join(repr(c) for c in self.cards)
    
    def __contains__(self):
        return card in self.cards
    
    def to_str(self):
        return f"{self.card1}{self.card2}{self.card3}{self.card4}{self.card5}"
