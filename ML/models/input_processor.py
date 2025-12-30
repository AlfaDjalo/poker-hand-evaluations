"""
Input processor components used by the model factory.

Processors are responsible for:
- unpacking raw model inputs
- constructing derived representations (e.g. combined hand + board)
- invoking the shared encoder
- returning a structured dictionary of embeddings for downstream heads

Processors intentionally contain no trainable parameters. They define
data flow and symmetry, not learned behavior.
"""

class SingleHandProcessor():
    """
    Input processor for single hand / board state models.

    This processor accepts a single input tensor containing both hand
    and board channels, constructs derived representations, and produces
    embeddings using a shared encoder.

    Expected input:
        inputs: Tensor of shape (batch, 14, 4, 2)
            Channel 0 → hand cards
            Channel 1 → board cards

    Processing steps:
        - Split hand and board channels
        - Construct a combined representation (hand + board)
        - Encode hand, board, and combined tensors using the same encoder

    Output:
        dict[str, Tensor] with keys:
            - "hand_emb":   embedding of hand cards
            - "board_emb":  embedding of board cards
            - "combo_emb":  embedding of combined hand + board

    Notes:
        - The encoder is shared across all representations, ensuring
          consistent embedding space.
        - No parameters are owned by this class; it is purely structural.
    """
    def __init__(self, encoder):
        # Shared CardStateEncoder instance
        self.encoder = encoder

    def __call__(self, inputs, training=False):
        # Shared CardStateEncoder instance
        hand = inputs[..., 0:1]         # (batch, 14, 4, 1)
        board = inputs[..., 1:2]        # (batch, 14, 4, 1)

      # Combined representation used by some heads
        combo = hand + board

        # Encode all representations in a single encoder call
        hand_emb, board_emb, combo_emb = self.encoder([hand, board, combo], training=training)

        return {
            "hand_emb": hand_emb,
            "board_emb": board_emb,
            "combo_emb": combo_emb,
        }


class PairProcessor():
    """
    Input processor for pairwise comparison models.

    This processor accepts two independent hand/board inputs (A, B),
    applies identical preprocessing and encoding to both, and returns
    a structured dictionary of embeddings suitable for pairwise heads.

    Expected input:
        inputs: tuple(inputs_A, inputs_B)

        Each element is a Tensor of shape (batch, 14, 4, 2):
            Channel 0 → hand cards
            Channel 1 → board cards

    Processing steps (applied symmetrically to A and B):
        - Split hand and board channels
        - Construct combined (hand + board) representation
        - Encode hand, board, and combined tensors using a shared encoder

    Output:
        dict[str, Tensor] with keys:
            - "hand_emb_A", "board_emb_A", "combo_emb_A"
            - "hand_emb_B", "board_emb_B", "combo_emb_B"

    Notes:
        - The same encoder instance is reused for A and B, enforcing
          symmetry and comparable embedding spaces.
        - This processor does not impose any ordering or comparison logic;
          that responsibility is delegated to the downstream head.
    """
    def __init__(self, encoder):
        # Shared encoder ensures symmetric treatment of A and B
        self.encoder = encoder

    def __call__(self, inputs, training=False):
        # Unpack paired inputs
        inputs_A, inputs_B = inputs

        # Construct representations for input A
        hand_A = inputs_A[..., 0:1]         # (batch, 14, 4, 1)
        board_A = inputs_A[..., 1:2]        # (batch, 14, 4, 1)
        combo_A = hand_A + board_A

        # Construct representations for input B
        hand_B = inputs_B[..., 0:1]         # (batch, 14, 4, 1)
        board_B = inputs_B[..., 1:2]        # (batch, 14, 4, 1)
        combo_B = hand_B + board_B

        # Encode A and B independently but with shared weights
        hand_emb_A, board_emb_A, combo_emb_A = self.encoder([hand_A, board_A, combo_A], training=training)
        hand_emb_B, board_emb_B, combo_emb_B = self.encoder([hand_B, board_B, combo_B], training=training)

        # return ((hand_emb_A, board_emb_A, combo_emb_A), (hand_emb_B, board_emb_B, combo_emb_B))
        return {
            "hand_emb_A": hand_emb_A,
            "board_emb_A": board_emb_A,
            "combo_emb_A": combo_emb_A,
            "hand_emb_B": hand_emb_B,
            "board_emb_B": board_emb_B,
            "combo_emb_B": combo_emb_B,
        }
