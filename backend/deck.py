from card import Card
from typing import Callable

# Create static data
RANKS = '23456789TJQKA'
SUITS = 'shdc'

class Deck:
    def __init__(self):
        self.deck = []
        self.create_deck()

    def create_deck(self):
        for r in RANKS:
            for s in SUITS:
                self.deck.append(Card(r, s))
        return
    
    def __repr__(self):
        return ', '.join(repr(c) for c in self.deck)
        
    def get_cards(self):
        return self.deck
    
    def filter_cards(self, predicate: Callable[[Card], bool]):
        return [card for card in self.deck if predicate(card)]
    