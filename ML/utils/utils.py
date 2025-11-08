RANK_ORDER = {r: i for i, r in enumerate('AKQJT98765432')}
SUIT_ORDER = {'s': 0, 'h': 1, 'd': 2, 'c':3}
SUITS = list(SUIT_ORDER.keys())
RANKS = list(RANK_ORDER.keys())

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

