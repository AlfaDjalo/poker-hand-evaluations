import numpy as np
import tensorflow as tf
import random
import itertools

RANKS = 'AKQJT98765432'
SUITS = 'shdc'

def create_tensor_grids(pairwise=False, rows=None):
    """
    Function for converting database rows to model input data for absolute_value model.
    
    Args:
        mode (int): 0 - single hand; 1 - pair of hands
        rows (tuple): Database rows, tuples of (hand_mask, board_mask, high_value, low_value)

    Returns:
        tuple (x, y):
            x is (len(rows), 13, 4, 2) tensor of ([hand_grid], [board_grid])
            y is (len(rows)) tensor of floats [0, 1] for the output labels
    """

    def duplicate_ace_rank(grid):
        # grid: (13,4) -> (14,4)
        ace_row = grid[0:1]      # Rank 'A' at index 0
        return np.concatenate([ace_row, grid], axis=0)

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

        # ✅ NEW: duplicate the Ace row
        hand_grid  = duplicate_ace_rank(hand_grid)
        board_grid = duplicate_ace_rank(board_grid)
        
        combined = np.stack([hand_grid, board_grid], axis=-1)

        return combined
    

    if pairwise:
    # if mode in ["absolute_value", "embedding_value", "hand_category"]:
        inputs_A = []
        inputs_B = []
    else:
        inputs = []
    
    labels = []

    for row in rows:
        if pairwise:
        # if mode in ["absolute_value", "embedding_value", "hand_category"]:
            (hand_A, hand_B, board_A, board_B, high_value_A, high_value_B) = row
            grid_A = create_grids(hand_A, board_A)
            grid_B = create_grids(hand_B, board_B)

            inputs_A.append(grid_A)
            inputs_B.append(grid_B)
            
            if high_value_A == high_value_B:
                labels.append(0.5)
            else:
                labels.append(1.0 if high_value_A < high_value_B else 0.0)
        else:
            (hand_mask, board_mask, high_value, low_value) = row
            combined = create_grids(hand_mask, board_mask)
            inputs.append(combined)
            labels.append((high_value - 1) / 7461.0)

    if pairwise:
    # if mode in ["absolute_value", "embedding_value", "hand_category"]:
        x = (
            tf.convert_to_tensor(np.stack(inputs_A)),
            tf.convert_to_tensor(np.stack(inputs_B))
        )
    else:
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


def augment_batch_with_suit_permutations(batch_inputs):
    """
    batch_inputs: tensor of shape (batch_size, 13, 4, channels)
    returns: tensor of shape (batch_size * 24, 13, 4, channels)
    """
    perms = list(itertools.permutations(range(4)))
    batch_size = tf.shape(batch_inputs)[0]

    # Prepare a list to store all permuted batches
    permuted_batches = []

    for perm in perms:
        # permute suits by gathering on axis=2 (the suit axis)
        permuted = tf.gather(batch_inputs, indices=list(perm), axis=2)
        permuted_batches.append(permuted)

    # Stack all permutations along new axis: shape (24, batch_size, 13, 4, channels)
    permuted_batches = tf.stack(permuted_batches, axis=0)

    # Transpose to (batch_size, 24, 13, 4, channels)
    permuted_batches = tf.transpose(permuted_batches, perm=[1, 0, 2, 3, 4])
    
    # Merge batch and perm dimensions: (batch_size * 24, 13, 4, channels)
    permuted_batches = tf.reshape(permuted_batches, (batch_size * 24, 14, 4, batch_inputs.shape[-1]))
    # permuted_batches = tf.reshape(permuted_batches, (batch_size * 24, 13, 4, batch_inputs.shape[-1]))

    return permuted_batches

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
# Value generator (supervised regression)
# ==========================================================
class ValueGenerator(BaseGenerator):
    """Fetches absolute combo value samples for regression training."""
    def __init__(self, config):
        super().__init__(config)
        if not self.is_validation:
            self.x, self.y = self._load_new_batch()

    def _load_new_batch(self):
        """ Pull a new large batch from DB and convert to tensors. """
        sample_evaluations = self.db.get_sample_evaluations(self.db_batch_size)
        x_tensor, y_tensor = create_tensor_grids(False, sample_evaluations)
        # x_tensor, y_tensor = create_tensor_grids(self.mode, sample_evaluations)
        return x_tensor.numpy(), y_tensor.numpy()

    def __len__(self):
        if self.is_validation and self._preloaded_x is not None:
            return max(1, len(self._preloaded_x) // self.batch_size)
        return self.db_batch_size // self.batch_size
    
    def __getitem__(self, idx):
        if self.is_validation and self._preloaded_x is not None:
            # Slice pre-loaded validation data, no augmentation for validation
            start = (idx * self.batch_size) % len(self._preloaded_x)
            end = start + self.batch_size
            y_batch = self._preloaded_y[start:end]
            return self._preloaded_x[start:end], y_batch
            # return self._preloaded_x[start:end], (y_batch, y_batch, y_batch)

        if not hasattr(self, 'x'):
            self.x, self.y = self._load_new_batch()

        start = (idx * self.batch_size) % len(self.x)
        end = start + self.batch_size

        batch_x = self.x[start:end]
        batch_y = self.y[start:end]

        # Apply augmentation here (training only)
        batch_x_aug = augment_batch_with_suit_permutations(batch_x)
        batch_y_aug = np.repeat(batch_y, 24, axis=0) # repeat labels for augmentation batch

        return batch_x_aug, batch_y_aug
        return batch_x_aug, (batch_y_aug, batch_y_aug, batch_y_aug)

    def __getitem_old__(self, idx):
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

    def preload_validation_data(self):
        # Fetch a large validation batch from DB (no augmentation)
        sample_evaluations = self.db.get_sample_evaluations(self.db_batch_size)
        
        # Convert rows to tensors using existing helper
        x_tensor, y_tensor = create_tensor_grids(False, sample_evaluations)
        # x_tensor, y_tensor = create_tensor_grids(self.mode, sample_evaluations)
        
        # Cache the numpy arrays for slicing in __getitem__
        self._preloaded_x = x_tensor.numpy()
        self._preloaded_y = y_tensor.numpy()



# ==========================================================
# Category generator (supervised classification)
# ==========================================================
class HandCategoryGenerator(ValueGenerator):
    """
    Same as AbsoluteGenerator, but converts treys rank values
    into a categorical class {0..8}.
    """

    @staticmethod
    def rank_to_category(rank):
        if 1 <= rank <= 10:
            return 0                    # Straight Flush
        if 11 <= rank <= 166:
            return 1                    # Quads
        if 167 <= rank <= 322:
            return 2                    # Full House
        if 323 <= rank <= 1599:
            return 3                    # Flush
        if 1600 <= rank <= 1609:
            return 4                    # Straight
        if 1610 <= rank <= 2467:
            return 5                    # Trips
        if 2468 <= rank <= 3325:
            return 6                    # Two Pair
        if 3326 <= rank <= 6185:
            return 7                    # One Pair
        return 8                        # No Pair

    def _load_new_batch(self):
        """
        Loads the rank values from DB but converts them into
        categorical classes instead of regression targets.
        """
        sample_evaluations = self.db.get_sample_evaluations(self.db_batch_size)

        # Reuse your existing tensor builder
        x_tensor, rank_tensor = create_tensor_grids(False, sample_evaluations)
        # x_tensor, rank_tensor = create_tensor_grids(self.mode, sample_evaluations)

        # rank_tensor is shape (N, 1) normalized 0..1
        # convert back to treys rank (1..7461)
        rank_np = rank_tensor.numpy().reshape(-1)
        treys_rank = (rank_np * 7461).astype(int) + 1

        # Convert to category 0..8
        class_idx = np.array([self.rank_to_category(r) for r in treys_rank], dtype=np.int32)

        # Convert to one-hot (N, 9)
        category_onehot = tf.one_hot(class_idx, depth=9, dtype=tf.float32).numpy()

        # Return x and ONE-HOT y
        return x_tensor.numpy(), category_onehot

        # # rank_tensor is shape (N, 1) or (N,)
        # rank_np = rank_tensor.numpy().reshape(-1)

        # # Convert to category class (vectorized)
        # category_np = np.array([self.rank_to_category(r) for r in rank_np], dtype=np.int32)

        # # Model expects 3 identical heads
        # return x_tensor.numpy(), category_np

    def __getitem__(self, idx):
        if self.is_validation and self._preloaded_x is not None:
            start = (idx * self.batch_size) % len(self._preloaded_x)
            end = start + self.batch_size

            y_batch = self._preloaded_y[start:end]
            return self._preloaded_x[start:end], y_batch
            # return self._preloaded_x[start:end], (y_batch, y_batch, y_batch)

        if not hasattr(self, 'x'):
            self.x, self.y = self._load_new_batch()

        start = (idx * self.batch_size) % len(self.x)
        end = start + self.batch_size

        batch_x = self.x[start:end]
        batch_y = self.y[start:end]

        # Suit augmentation
        batch_x_aug = augment_batch_with_suit_permutations(batch_x)
        batch_y_aug = np.repeat(batch_y, 24, axis=0)

        return batch_x_aug, batch_y_aug
        # return batch_x_aug, (batch_y_aug, batch_y_aug, batch_y_aug)

    def preload_validation_data(self):
        """
        Load validation batch and convert rank values → one-hot categories.
        """
        sample_evaluations = self.db.get_sample_evaluations(self.db_batch_size)
        x_tensor, rank_tensor = create_tensor_grids(False, sample_evaluations)
        # x_tensor, rank_tensor = create_tensor_grids(self.mode, sample_evaluations)

        rank_np = rank_tensor.numpy().reshape(-1)
        # Convert normalized rank back to treys rank (1..7461)
        treys_rank = (rank_np * 7461).astype(int) + 1
        # Convert to category class (0..8)
        class_idx = np.array([self.rank_to_category(r) for r in treys_rank], dtype=np.int32)
        # Convert to one-hot (N, 9)
        category_onehot = tf.one_hot(class_idx, depth=9, dtype=tf.float32).numpy()

        self._preloaded_x = x_tensor.numpy()
        self._preloaded_y = category_onehot


# ==========================================================
# Pairwise generator (supervised regression)
# ==========================================================
class PairwiseComparisonGenerator(BaseGenerator):
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
        x_tensor, y_tensor = create_tensor_grids(True, sample_evaluations)
        # x_tensor, y_tensor = create_tensor_grids(self.mode, sample_evaluations)
        return x_tensor, y_tensor

    def __len__(self):
        if self.is_validation and self._preloaded_x is not None:
            return max(1, len(self._preloaded_x[0]) // self.batch_size)
        return self.db_batch_size // self.batch_size
        
    def __getitem__(self, idx):
        if self.is_validation and self._preloaded_x is not None:
            # Validation mode: no augmentation
            start = (idx * self.batch_size) % len(self._preloaded_x[0])
            end = start + self.batch_size
            
            x_A = self._preloaded_x[0][start:end]
            x_B = self._preloaded_x[1][start:end]
            y_batch = self._preloaded_y[start:end]
            
            return (x_A, x_B), (y_batch, y_batch, y_batch)
        
        # Training mode: load fresh batch if needed
        if not hasattr(self, 'x'):
            self.x, self.y = self._load_new_batch()
        
        start = (idx * self.batch_size) % len(self.x[0])
        end = start + self.batch_size
        
        x_A = self.x[0][start:end]  # shape (batch_size, 13, 4, channels)
        x_B = self.x[1][start:end]
        y_batch = self.y[start:end]

        # Augment inputs with suit permutations
        x_A_aug = augment_batch_with_suit_permutations(x_A)  # (batch_size*24, 13, 4, channels)
        x_B_aug = augment_batch_with_suit_permutations(x_B)

        # Repeat labels for augmented batch size
        y_batch_aug = np.repeat(y_batch, 24, axis=0)

        return (x_A_aug, x_B_aug), (y_batch_aug, y_batch_aug, y_batch_aug)

    def __getitem_old__(self, idx):
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

    def preload_validation_data(self):
        # Always validate on 'mix'
        val_mode = "mix"
        sample_evaluations = self.db.get_comparison_pairs(val_mode, self.db_batch_size)
        x_tensor, y_tensor = create_tensor_grids(True, sample_evaluations)
        # x_tensor, y_tensor = create_tensor_grids(val_mode, sample_evaluations)

        self._preloaded_x = (x_tensor[0].numpy(), x_tensor[1].numpy())
        self._preloaded_y = y_tensor.numpy()


# ==========================================================
# Alternating generator (mixes pair types)
# ==========================================================
class AlternatingGenerator(PairwiseComparisonGenerator):
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
        x_tensor, y_tensor = create_tensor_grids(True, sample_evaluations)
        # x_tensor, y_tensor = create_tensor_grids(self.current_mode, sample_evaluations)
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
            
            return (x_A, x_B), y_batch
            return (x_A, x_B), (y_batch, y_batch, y_batch)
        
        # Training mode: load fresh batch each call
        if not hasattr(self, 'x'):
            self.x, self.y = self._load_new_batch()
        
        start = (idx * self.batch_size) % len(self.x[0])
        end = start + self.batch_size
        
        x_A = self.x[0][start:end]
        x_B = self.x[1][start:end]
        y_batch = self.y[start:end]
        
        return (x_A, x_B), y_batch
        # return (x_A, x_B), (y_batch, y_batch, y_batch)

    def on_epoch_end(self):
        """Reload data with next mode in cycle (training only)."""
        if not self.is_validation:
            self.current_mode = next(self.cycle) if not self.mix_ratio else np.random.choice(["board", "hand", "mix"], p=self.mix_ratio)
            self.x, self.y = self._load_new_batch()


class AdaptedGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_generator, adapter):
        self.base = base_generator
        self.adapter = adapter

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return self.adapter.adapt(x, y)

    def on_epoch_end(self):
        if hasattr(self.base, "on_epoch_end"):
            self.base.on_epoch_end()

    def __getattr__(self, name):
        return getattr(self.base, name)
    

class IdentityAdapter:
    def adapt(self, x, y):
        return x, y

class RepeatTargetAdapter:
    def __init__(self, count):
        self.count = count

    def adapt(self, x, y):
        return x, tuple(y for _ in range(self.count))

def build_output_adapter(output_adapter):
    type = output_adapter.get("type", "identity")
    count = output_adapter.get("count", 1)
    if type == "identity":
        return IdentityAdapter()
    elif type == "repeat":
        return RepeatTargetAdapter(count)

    raise ValueError(f"Unknown output adapter: {type}")