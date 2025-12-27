

class SingleHandProcessor():
    """
    Processes a single hand input.
    """
    def __init__(self, encoder):
        self.encoder = encoder

    def __call__(self, inputs, training=False):
        hand = inputs[..., 0:1]         # (batch, 14, 4, 1)
        board = inputs[..., 1:2]        # (batch, 14, 4, 1)
        combo = hand + board

        hand_emb, board_emb, combo_emb = self.encoder([hand, board, combo], training=training)

        return {
            "hand_emb": hand_emb,
            "board_emb": board_emb,
            "combo_emb": combo_emb,
        }

class PairProcessor():
    """
    Processes paired hand inputs (A, B) using a shared encoder.
    """
    def __init__(self, encoder):
        self.encoder = encoder

    def __call__(self, inputs, training=False):
        inputs_A, inputs_B = inputs

        hand_A = inputs_A[..., 0:1]         # (batch, 14, 4, 1)
        board_A = inputs_A[..., 1:2]        # (batch, 14, 4, 1)
        combo_A = hand_A + board_A

        hand_B = inputs_B[..., 0:1]         # (batch, 14, 4, 1)
        board_B = inputs_B[..., 1:2]        # (batch, 14, 4, 1)
        combo_B = hand_B + board_B

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
