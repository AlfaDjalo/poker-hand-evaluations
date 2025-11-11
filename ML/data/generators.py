import numpy as np
import tensorflow as tf
import random

RANKS = 'AKQJT98765432'
SUITS = 'shdc'

def create_tensor_grids(rows):
    inputs = []
    labels = []

    for (hand_mask, board_mask, high_value, low_value) in rows:
        hand_grid = np.zeros((13,4), dtype=np.float32)
        board_grid = np.zeros((13,4), dtype=np.float32)
        # full_grid = np.zeros((13,4), dtype=np.float32)

        for bit_index in range(52):
            rank = bit_index // 4
            suit = bit_index % 4
            if hand_mask & (1 << bit_index):
                hand_grid[rank][suit] = 1
                # full_grid[rank][suit] = 1
            if board_mask & (1 << bit_index):
                board_grid[rank][suit] = 1
                # full_grid[rank][suit] = 1

        combined = np.stack([hand_grid, board_grid], axis=-1)
        # combined = np.stack([hand_grid, board_grid, full_grid], axis=-1)
        inputs.append(combined)
        labels.append((high_value - 1) / 7461.0)
        # labels.append(high_value)

    x = tf.convert_to_tensor(np.stack(inputs))
    y = tf.convert_to_tensor(np.array(labels), dtype=tf.float32)[..., None]
    return x, y


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


class BaseGenerator(tf.keras.utils.Sequence):
    def __init__(self, db, batch_size, mode):
        self.db = db
        self.batch_size = batch_size
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == "board":
            return self._board_batch()
        elif self.mode == "hand":
            return self._hand_batch()
        elif self.mode == "mixed":
            return self._mixed_batch()
        
    def _board_batch(self):
        pass

    def _hand_batch(self):
        pass

    def _mixed_batch(self):
        pass


class AlternatingGenerator(tf.keras.utils.Sequence):
    def __init__(self, db, batch_size, probs=(0.4, 0.4, 0.2)):
        self.db = db
        self.batch_size = batch_size
        self.probs = probs

    def __iter__(self):
        while True:
            mode = random.choices(["board", "hand", "mixed"], self.probs)[0]
            yield BaseGenerator(self.db, self.batch_size, mode).__getitem__(0)


class AbsoluteGenerator(tf.keras.utils.Sequence):
    def __init__(self, config):
        self.db = config["db"]
        self.db_batch_size = config.get("db_batch_size", 20000)
        self.model_batch_size = config.get("model_batch_size", 1024)

        self.x, self.y = self._load_new_batch()

    def _load_new_batch(self):
        """ Pull a new large batch from DB and convert to tensors. """
        sample_evaluations = self.db.get_sample_evaluations(self.db_batch_size)
        x_tensor, y_tensor = create_tensor_grids(sample_evaluations)
        return x_tensor.numpy(), y_tensor.numpy()

    def __len__(self):
        return self.db_batch_size // self.model_batch_size
    
    def __getitem__(self, idx):
        """ Return one sub-batch. """
        # rows = self.db.get_sample_evaluations(self.db_batch_size)
        # x, y = create_tensor_grids(rows)
        # Select a slice to yield this step
        start = (idx * self.model_batch_size) % len(self.x)
        end = start + self.model_batch_size
        y_batch = self.y[start:end]
        return self.x[start:end], (y_batch, y_batch, y_batch)
        # return self.x[start:end], self.y[start:end]

    def on_epoch_end(self):
        """ Reload data at end of each epoch. """
        self.x, self.y = self._load_new_batch()

class PairwiseGenerator(tf.keras.utils.Sequence):
    def __init__(self, config, db=None):
        self.db = db or config["db"]
        self.db_batch_size = config.get("db_batch_size", 20000)
        self.model_batch_size = config.get("model_batch_size", 1024)
        self.same_board_fraction = config.get("same_board_fractions", 0.5)

    def __len__(self):
        return self.db_batch_size // self.model_batch_size
    
    def __getitem__(self, idx):
        rows = self.db.get_sample_evaluations(self.db_batch_size)
        x, y = create_tensor_grids(rows)

        pairs_x1, pairs_x2, pairs_y = [], [], []

        for _ in range(self.model_batch_size):
            i, j = random.sample(range(len(x)), 2)
            x1, x2 = x[i], x[j]
            label = 1.0 if y[i] > y[j] else 0.0
            pairs_x1.append(x1)
            pairs_x2.append(x2)
            pairs_y.append(label)

        return [tf.stack(pairs_x1), tf.stack(pairs_x2)], tf.convert_to_tensor(pairs_y, dtype=tf.float32)

