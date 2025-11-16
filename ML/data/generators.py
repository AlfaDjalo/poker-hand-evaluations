import numpy as np
import tensorflow as tf
import random
import itertools

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

def create_pairwise_tensor_grids(hands_a, hands_b, boards_a, boards_b):
    """
    Convert paired hand/board bitmasks into model input format.
    
    Returns:
        x1: tuple of (hand1, board1, combo1) tensors
        x2: tuple of (hand2, board2, combo2) tensors
    """
    def bitmask_to_grids(hand_mask, board_mask):
        """Convert single hand/board bitmask pair to grids."""
        hand_grid = np.zeros((13, 4), dtype=np.float32)
        board_grid = np.zeros((13, 4), dtype=np.float32)

        for bit_index in range(52):
            rank = bit_index // 4
            suit = bit_index % 4
            if hand_mask & (1 << bit_index):
                hand_grid[rank][suit] = 1
            if board_mask & (1 << bit_index):
                board_grid[rank][suit] = 1

        return hand_grid, board_grid

    hands1_grids = []
    boards1_grids = []
    combos1_grids = []

    hands2_grids = []
    boards2_grids = []
    combos2_grids = []
    
    for hand_a, board_a, hand_b, board_b in zip(hands_a, boards_a, hands_b, boards_b):
        hand1_grid, board1_grid = bitmask_to_grids(hand_a, board_a)
        combo1_grid = hand1_grid + board1_grid
        # combo1_grid = np.stack([hand1_grid, board1_grid, hand1_grid+board1_grid], axis=-1)

        hands1_grids.append(hand1_grid[..., None])
        boards1_grids.append(board1_grid[..., None])
        combos1_grids.append(combo1_grid[..., None])
        # combos1_grids.append(combo1_grid)

        hand2_grid, board2_grid = bitmask_to_grids(hand_b, board_b)
        combo2_grid = hand2_grid + board2_grid
        # combo2_grid = np.stack([hand2_grid, board2_grid, hand2_grid+board2_grid], axis=-1)

        hands2_grids.append(hand2_grid[..., None])
        boards2_grids.append(board2_grid[..., None])
        combos2_grids.append(combo2_grid[..., None])
        # combos2_grids.append(combo2_grid)

    x1 = (
        tf.convert_to_tensor(np.stack(hands1_grids)),
        tf.convert_to_tensor(np.stack(boards1_grids)),
        tf.convert_to_tensor(np.stack(combos1_grids))        
    )    

    x2 = (
        tf.convert_to_tensor(np.stack(hands2_grids)),
        tf.convert_to_tensor(np.stack(boards2_grids)),
        tf.convert_to_tensor(np.stack(combos2_grids))        
    )    

    return x1, x2



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

    # def __init__(self, db, batch_size, mode):
        # self.db = db
        # self.batch_size = batch_size
        # self.mode = mode

    def __len__(self):
        return 100 # unused ?

    def __getitem__(self, idx):
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
        self.x, self.y = self._load_new_batch()

    def _load_new_batch(self):
        """ Pull a new large batch from DB and convert to tensors. """
        sample_evaluations = self.db.get_sample_evaluations(self.db_batch_size)
        x_tensor, y_tensor = create_tensor_grids(sample_evaluations)
        return x_tensor.numpy(), y_tensor.numpy()

    def __len__(self):
        return self.db_batch_size // self.batch_size
    
    def __getitem__(self, idx):
        """ Return one sub-batch. """
        # rows = self.db.get_sample_evaluations(self.db_batch_size)
        # x, y = create_tensor_grids(rows)
        # Select a slice to yield this step
        start = (idx * self.batch_size) % len(self.x)
        end = start + self.batch_size
        y_batch = self.y[start:end]
        return self.x[start:end], (y_batch, y_batch, y_batch)
        # return self.x[start:end], self.y[start:end]

    def on_epoch_end(self):
        """ Reload data at end of each epoch. """
        self.x, self.y = self._load_new_batch()


# ==========================================================
# Pairwise generator (ranking training)
# ==========================================================
class PairwiseGenerator(BaseGenerator):
    """Generates (x1, x2, y) pairs for pairwise / ranking training."""
    def __init__(self, config, mode):
        super().__init__(config)
        self.mode = mode
        self.x1, self.x2, self.y = self._load_new_batch()

    def _load_new_batch(self):
        """Pull a new large batch from DB and convert to tensors."""
        print("⏳ Querying DB...")
        rows = self.db.get_comparison_pairs(self.mode, self.db_batch_size)
        print(f"✅ DB returned {len(rows)} rows")
        if rows:
            print("Example row:", rows[0])

        hands_a, hands_b, boards_a, boards_b, values_a, values_b = self.unpack_rows(rows)

        x1, x2 = create_pairwise_tensor_grids(hands_a, hands_b, boards_a, boards_b)

        y = (values_a > values_b).astype("float32")
        y_tensor = tf.convert_to_tensor(y)[..., None]

        return (
            (x1[0].numpy(), x1[1].numpy(), x1[2].numpy()),
            (x2[0].numpy(), x2[1].numpy(), x2[2].numpy()),
            y_tensor.numpy()
        )    

    def __len__(self):
        return self.db_batch_size // self.batch_size
    
    def __getitem__(self, idx):
        """Return one sub-batch."""
        start = (idx * self.batch_size) % len(self.y)
        end = start + self.batch_size
        
        # Extract slices for this batch
        x1_batch = (
            self.x1[0][start:end],
            self.x1[1][start:end],
            self.x1[2][start:end]
        )
        
        x2_batch = (
            self.x2[0][start:end],
            self.x2[1][start:end],
            self.x2[2][start:end]
        )
        
        y_batch = self.y[start:end]
        
        # Return format: (x1, x2), (y, y, y) for three outputs
        return (x1_batch, x2_batch), (y_batch, y_batch, y_batch)

    def on_epoch_end(self):
        """Reload data at end of each epoch."""
        self.x1, self.x2, self.y = self._load_new_batch()

    # def __getitem__(self, idx):
    #     print("⏳ Querying DB...")
    #     rows = self.db.get_comparison_pairs(self.mode, self.db_batch_size)
    #     print("Example row:", rows[0])
    #     print("✅ DB returned", len(rows), "rows")

    #     hands_a, hands_b, boards_a, boards_b, values_a, values_b = self.unpack_rows(rows)

    #     x1 = self.encoder.encode(hands_a, boards_a)
    #     x2 = self.encoder.encode(hands_b, boards_b)

    #     y = (values_a > values_b).astype("float32")

    #     return (x1, x2), (y, y, y)

        # return rows

    # def __getitem__(self, idx):
    #     """Ask DB for comparison pairs according to current mode."""
    #     return self.db.get_comparison_pairs(self.mode, self.db_batch_size)
  

# ==========================================================
# Alternating generator (mixes pair types)
# ==========================================================
class AlternatingGenerator(PairwiseGenerator):
    """Mixes 'board' and 'hand' pair types dynamically."""
    def __init__(self, config, cycle=("board", "hand")):
        BaseGenerator.__init__(self, config)  # Call grandparent directly
        # super().__init__(config, mode="alternating")
        self.mode = "alternating"
        self.cycle = itertools.cycle(cycle)
        self.mix_ratio = config.get("mix_ratio", None)

        self.current_mode = next(self.cycle) if not self.mix_ratio else np.random.choice(["board", "hand", "mix"], p=self.mix_ratio)
        self.x1, self.x2, self.y = self._load_batch_with_mode(self.current_mode)

        def __getitem__(self, idx):
            """Return one sub-batch."""
            start = (idx * self.batch_size) % len(self.y)
            end = start + self.batch_size
            
            x1_batch = (
                self.x1[0][start:end],
                self.x1[1][start:end],
                self.x1[2][start:end]
            )
            
            x2_batch = (
                self.x2[0][start:end],
                self.x2[1][start:end],
                self.x2[2][start:end]
            )
            
            y_batch = self.y[start:end]
            
            return (x1_batch, x2_batch), (y_batch, y_batch, y_batch)

    # def __getitem__(self, idx):
    #     mode = next(self.cycle)

    #     if self.mix_ratio is not None:
    #         mode = np.random.choice(["board", "hand", "mix"], p=self.mix_ratio)

    #     return self.db.get_comparison_pairs(mode, self.db_batch_size)

    def _load_batch_with_mode(self, mode):
        """Load batch using specific mode."""
        print("⏳ Querying DB...")
        rows = self.db.get_comparison_pairs(mode, self.db_batch_size)
        print(f"✅ DB returned {len(rows)} rows")
        if rows:
            print("Example row:", rows[0])

        hands_a, hands_b, boards_a, boards_b, values_a, values_b = self.unpack_rows(rows)

        x1, x2 = create_pairwise_tensor_grids(hands_a, hands_b, boards_a, boards_b)

        y = (values_a > values_b).astype("float32")
        y_tensor = tf.convert_to_tensor(y)[..., None]

        return (
            (x1[0].numpy(), x1[1].numpy(), x1[2].numpy()),
            (x2[0].numpy(), x2[1].numpy(), x2[2].numpy()),
            y_tensor.numpy()
        )    

    def on_epoch_end(self):
        """Reload data with next mode in cycle."""
        self.current_mode = next(self.cycle) if not self.mix_ratio else np.random.choice(["board", "hand", "mix"], p=self.mix_ratio)
        self.x1, self.x2, self.y = self._load_batch_with_mode(self.current_mode)
    # def __iter__(self):
    #     while True:
    #         mode = random.choices(["board", "hand", "mixed"], self.probs)[0]
    #         yield BaseGenerator(self.db, self.batch_size, mode).__getitem__(0)



