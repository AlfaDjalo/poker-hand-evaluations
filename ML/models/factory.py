import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from .encoders import CardStateEncoder, get_encoder_config
from .heads import EmbeddingValueHead, PairwiseEmbeddingHead, CategoryEmbeddingHead
from typing import Dict

# Simple mapping so mode_config can pass a head_type string.
_HEAD_MAP = {
    "pairwise": PairwiseEmbeddingHead,
    "value": EmbeddingValueHead,
    "category": CategoryEmbeddingHead,
}

def _slice_hand_board_combo(inp):
    hand = layers.Lambda(lambda x: x[..., 0:1], name="slice_hand")(inp)
    board = layers.Lambda(lambda x: x[..., 1:2], name="slice_board")(inp)
    combo = layers.Add(name="combine_hand_board")([hand, board])
    return hand, board, combo

def build_model(config: Dict, mode_cfg: Dict):
    """
    Generic factory. mode_cfg must contain:
      - head_type: "pairwise" | "value" | "category"
      - generator / other training fields are unchanged
    """
    head_type = mode_cfg.get("head_type", "value")
    head_cls = _HEAD_MAP[head_type]

    encoder_cfg = get_encoder_config(config)
    encoder = CardStateEncoder(encoder_cfg)

    if head_type == "pairwise":
        # Two-grid (A/B) inputs -> embeddings for A and B -> three pairwise heads (hand, board, combined)
        in_a = Input(shape=config["input_shape"], name="input_A")
        in_b = Input(shape=config["input_shape"], name="input_B")

        hand_a, board_a, combo_a = _slice_hand_board_combo(in_a)
        hand_b, board_b, combo_b = _slice_hand_board_combo(in_b)

        hand_emb_a, board_emb_a, combo_emb_a = encoder([hand_a, board_a, combo_a])
        hand_emb_b, board_emb_b, combo_emb_b = encoder([hand_b, board_b, combo_b])

        hand_head = head_cls()( [hand_emb_a, hand_emb_b] )
        board_head = head_cls()( [board_emb_a, board_emb_b] )
        combined_head = head_cls()( [combo_emb_a, combo_emb_b] )

        model = Model(inputs=[in_a, in_b], outputs=[hand_head, board_head, combined_head])
        return model

    elif head_type in ("value", "category"):
        # Single input -> encoder -> embedding heads (produce either scalar(s) or categories)
        inp = Input(shape=config["input_shape"], name="poker_input")
        hand, board, combo = _slice_hand_board_combo(inp)
        hand_emb, board_emb, combo_emb = encoder([hand, board, combo])

        if head_type == "value":
            # produce three value heads (hand/board/combined)
            hand_v = EmbeddingValueHead()(hand_emb)
            board_v = EmbeddingValueHead()(board_emb)
            combined_v = EmbeddingValueHead()(combo_emb)
            return Model(inputs=inp, outputs=[hand_v, board_v, combined_v])

        else:
            # category head (softmax) for hand/board/combined
            hand_c = CategoryEmbeddingHead()(hand_emb)
            board_c = CategoryEmbeddingHead()(board_emb)
            combined_c = CategoryEmbeddingHead()(combo_emb)
            return Model(inputs=inp, outputs=[hand_c, board_c, combined_c])

    else:
        raise ValueError(f"Unsupported head_type: {head_type}")
