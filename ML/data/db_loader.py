import numpy as np
import torch
# import pandas as pd
# import json
# import tensorflow as tf

import sys
import os

# Add the back-end folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from db_plo import DB_PLO, open_db

RANK_ORDER = {r: i for i, r in enumerate('AKQJT98765432')}
SUIT_ORDER = {'s': 0, 'h': 1, 'd': 2, 'c':3}
SUITS = list(SUIT_ORDER.keys())
RANKS = list(RANK_ORDER.keys())

def main():

    db = open_db()
    gen = data_generator(db, db_batch_size=10, model_batch_size=3)

    for step, (x, y) in enumerate(gen):
        print("Step: ", step)
        print("(x, y): ", x, y)


def create_tensor_grids(rows):
    inputs = []
    labels = []

    for (hand_mask, board_mask, high_value, low_value) in rows:
        hand_grid = np.zeros((13,4), dtype=np.float32)
        board_grid = np.zeros((13,4), dtype=np.float32)
        full_grid = np.zeros((13,4), dtype=np.float32)

        for bit_index in range(52):
            if hand_mask & (1 << bit_index):
                rank = bit_index // 4
                suit = bit_index % 4
                hand_grid[rank][suit] = 1
                full_grid[rank][suit] = 1
            if board_mask & (1 << bit_index):
                rank = bit_index // 4
                suit = bit_index % 4
                board_grid[rank][suit] = 1
                full_grid[rank][suit] = 1

        combined = np.stack([hand_grid, board_grid, full_grid], axis=-1)
        inputs.append(combined)
        labels.append((high_value - 1) / 7461.0)
        # labels.append(high_value)

    x_tensor = torch.tensor(np.stack(inputs))
    y_tensor = torch.tensor(np.array(labels)).unsqueeze(1).float()
    return x_tensor, y_tensor


def data_generator(db, db_batch_size=20000, model_batch_size=1024):
    while True:
        sample_evaluations = db.get_sample_evaluations(db_batch_size)
        x_tensor, y_tensor = create_tensor_grids(sample_evaluations)

        for i in range(0, len(x_tensor), model_batch_size):
            yield x_tensor[i:i + model_batch_size], y_tensor[i:i+model_batch_size]


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


if __name__ == "__main__":
    main()
