import numpy as np
import tensorflow as tf
import random
import itertools

RANKS = 'AKQJT98765432'
SUITS = 'shdc'

def create_tensor_grids(mode, rows):
    """
    Function for converting database rows to model input data for absolute_value model.
    
    Args:
        mode (str): Training mode
        rows (tuple): Database rows, tuples of (hand_mask, board_mask, high_value, low_value)

    Returns:
        tuple (x, y):
            x is (len(rows), 13, 4, 2) tensor of ([hand_grid], [board_grid])
            y is (len(rows)) tensor of floats [0, 1] for the output labels
    """

    def create_grids(hand_mask, board_mask):
        """
        Helper function to create hand and board grids.
        
        Args:
            hand ():
            board ():

        Returns:
            tensor: Combined (hand, board) grids
        """
        hand_grid = np.zeros((13,4), dtype=np.float32)
        board_grid = np.zeros((13,4), dtype=np.float32)

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

        return combined
    

    if mode == "absolute_value":
        inputs = []
    else:
        inputs_A = []
        inputs_B = []
    
    labels = []

    for row in rows:
        if mode == "absolute_value":
            (hand_mask, board_mask, high_value, low_value) = row
            combined = create_grids(hand_mask, board_mask)
            inputs.append(combined)
            labels.append((high_value - 1) / 7461.0)
        else:
            (hand_A, hand_B, board_A, board_B, high_value_A, high_value_B) = row
            grid_A = create_grids(hand_A, board_A)
            grid_B = create_grids(hand_B, board_B)

            inputs_A.append(grid_A)
            inputs_B.append(grid_B)
            
            if high_value_A == high_value_B:
                labels.append(0.5)
            else:
                labels.append(1.0 if high_value_A < high_value_B else 0.0)

    if mode == "absolute_value":
        x = tf.convert_to_tensor(np.stack(inputs))
    else:
        x = (
            tf.convert_to_tensor(np.stack(inputs_A)),
            tf.convert_to_tensor(np.stack(inputs_B))
        )

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


# ==========================================================
# Base class
# ==========================================================
class BaseGenerator(tf.keras.utils.Sequence):
    """Base generator handling database access and batching."""
    def __init__(self, config):
        self.db = config["db"]
        self.db_batch_size = config.get("db_batch_size", 20000)
        self.batch_size = config.get("model_batch_size", 1024)
        self.mode = config.get("mode", "absolute_value")
        self.modes = ["board", "hand", "mix"]
        self.mode_index = 0
        self.is_validation = config.get("is_validation", False)  # ✅ NEW FLAG
        self._preloaded_x = None  # ✅ Cache for validation data
        self._preloaded_y = None

    # def __init__(self, db, batch_size, mode):
        # self.db = db
        # self.batch_size = batch_size
        # self.mode = mode

    def __len__(self):
        return 100 # unused ?

    def __getitem__(self, idx):
        if self.is_validation and self._preloaded_x is not None:
            # ✅ Use pre-loaded validation data, slice it
            start = (idx * self.batch_size) % len(self._preloaded_x)
            end = start + self.batch_size
            return self._preloaded_x[start:end], self._preloaded_y[start:end]
        
        if self.mode in ["board", "hand", "mix"]:
            return self.db.get_comparison_pairs(self.mode, self.db_batch_size)
        elif self.mode == "alternating":
            mode_to_use = self.modes[self.mode_index]
            self.mode_index = (self.mode_index + 1) % len(self.modes)
            return self.db.get_comparison_pairs(mode_to_use, self.db_batch_size)
        else:
            return self.db.get_sample_evaluations(self.db_batch_size)
    
    def unpack_rows(self, rows):
        hands_a, hands_b, boards_a, boards_b, values_a, values_b =zip(*rows)
        return np.array(hands_a), np.array(hands_b), np.array(boards_a), np.array(boards_b), np.array(values_a), np.array(values_b)

    def encode_inputs(self, hands, boards):
        return self.encoder.encode(hands, boards)


# ==========================================================
# Absolute value generator (supervised regression)
# ==========================================================
class AbsoluteGenerator(BaseGenerator):
    """Fetches absolute combo value samples for regression training."""
    def __init__(self, config):
        super().__init__(config)
        if not self.is_validation:
            self.x, self.y = self._load_new_batch()

    def _load_new_batch(self):
        """ Pull a new large batch from DB and convert to tensors. """
        sample_evaluations = self.db.get_sample_evaluations(self.db_batch_size)
        x_tensor, y_tensor = create_tensor_grids(self.mode, sample_evaluations)
        return x_tensor.numpy(), y_tensor.numpy()

    def __len__(self):
        if self.is_validation and self._preloaded_x is not None:
            return max(1, len(self._preloaded_x) // self.batch_size)
        return self.db_batch_size // self.batch_size
    
    def __getitem__(self, idx):
        if self.is_validation and self._preloaded_x is not None:
            # ✅ Slice pre-loaded validation data
            start = (idx * self.batch_size) % len(self._preloaded_x)
            end = start + self.batch_size
            y_batch = self._preloaded_y[start:end]
            return self._preloaded_x[start:end], (y_batch, y_batch, y_batch)
        
        # Training mode: load fresh batch each call
        if not hasattr(self, 'x'):
            self.x, self.y = self._load_new_batch()
        
        start = (idx * self.batch_size) % len(self.x)
        end = start + self.batch_size
        y_batch = self.y[start:end]
        return self.x[start:end], (y_batch, y_batch, y_batch)

    def on_epoch_end(self):
        """ Reload data at end of each epoch (training only). """
        if not self.is_validation:
            self.x, self.y = self._load_new_batch()


# ==========================================================
# Pairwise generator (supervised regression)
# ==========================================================
class PairwiseGenerator(BaseGenerator):
    """Fetches pairwise comparison samples for training."""
    def __init__(self, config, mode_override=None):
        super().__init__(config)
        if mode_override != None:
            self.mode = mode_override
        if not self.is_validation:
            self.x, self.y = self._load_new_batch()

    def _load_new_batch(self):
        """ Pull a new large batch from DB and convert to tensors. """
        sample_evaluations = self.db.get_comparison_pairs(self.mode, self.db_batch_size)
        x_tensor, y_tensor = create_tensor_grids(self.mode, sample_evaluations)
        return x_tensor, y_tensor

    def __len__(self):
        if self.is_validation and self._preloaded_x is not None:
            return max(1, len(self._preloaded_x[0]) // self.batch_size)
        return self.db_batch_size // self.batch_size
    
    def __getitem__(self, idx):
        if self.is_validation and self._preloaded_x is not None:
            # ✅ Slice pre-loaded validation data
            start = (idx * self.batch_size) % len(self._preloaded_x[0])
            end = start + self.batch_size
            
            x_A = self._preloaded_x[0][start:end]
            x_B = self._preloaded_x[1][start:end]
            y_batch = self._preloaded_y[start:end]
            
            return (x_A, x_B), (y_batch, y_batch, y_batch)
        
        # Training mode: load fresh batch each call
        if not hasattr(self, 'x'):
            self.x, self.y = self._load_new_batch()
        
        start = (idx * self.batch_size) % len(self.x[0])
        end = start + self.batch_size
        
        x_A = self.x[0][start:end]
        x_B = self.x[1][start:end]
        y_batch = self.y[start:end]
        
        return (x_A, x_B), (y_batch, y_batch, y_batch)

    def on_epoch_end(self):
        """ Reload data at end of each epoch (training only). """
        if not self.is_validation:
            self.x, self.y = self._load_new_batch()


# ==========================================================
# Alternating generator (mixes pair types)
# ==========================================================
class AlternatingGenerator(PairwiseGenerator):
    """Mixes 'board' and 'hand' pair types dynamically."""
    def __init__(self, config, cycle=("board", "hand", "mix")):
        BaseGenerator.__init__(self, config)
        self.mode = "alternating"
        self.cycle = itertools.cycle(cycle)
        self.mix_ratio = config.get("mix_ratio", None)

        self.current_mode = next(self.cycle) if not self.mix_ratio else np.random.choice(["board", "hand", "mix"], p=self.mix_ratio)
        if not self.is_validation:
            self.x, self.y = self._load_new_batch()

    def _load_new_batch(self):
        """ Pull a new large batch from DB and convert to tensors. """
        sample_evaluations = self.db.get_comparison_pairs(self.current_mode, self.db_batch_size)
        x_tensor, y_tensor = create_tensor_grids(self.current_mode, sample_evaluations)
        return x_tensor, y_tensor

    def __len__(self):
        if self.is_validation and self._preloaded_x is not None:
            return max(1, len(self._preloaded_x[0]) // self.batch_size)
        return self.db_batch_size // self.batch_size
    
    def __getitem__(self, idx):
        if self.is_validation and self._preloaded_x is not None:
            # ✅ Slice pre-loaded validation data
            start = (idx * self.batch_size) % len(self._preloaded_x[0])
            end = start + self.batch_size
            
            x_A = self._preloaded_x[0][start:end]
            x_B = self._preloaded_x[1][start:end]
            y_batch = self._preloaded_y[start:end]
            
            return (x_A, x_B), (y_batch, y_batch, y_batch)
        
        # Training mode: load fresh batch each call
        if not hasattr(self, 'x'):
            self.x, self.y = self._load_new_batch()
        
        start = (idx * self.batch_size) % len(self.x[0])
        end = start + self.batch_size
        
        x_A = self.x[0][start:end]
        x_B = self.x[1][start:end]
        y_batch = self.y[start:end]
        
        return (x_A, x_B), (y_batch, y_batch, y_batch)

    def on_epoch_end(self):
        """Reload data with next mode in cycle (training only)."""
        if not self.is_validation:
            self.current_mode = next(self.cycle) if not self.mix_ratio else np.random.choice(["board", "hand", "mix"], p=self.mix_ratio)
            self.x, self.y = self._load_new_batch()



