import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable
from .encoders import *

@register_keras_serializable(package="Poker")
class CombinedInputValueHead(Model):
    """
    Value head that operates directly on combined embedding.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value_head = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        return self.value_head(inputs)

    def get_config(self):
        return super().get_config()
    

@register_keras_serializable(package="Poker")
class SeparateInputValueHead(Model):
    """
    Value head that operates directly on hand and board embedding.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value_head = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        stacked_emb = tf.keras.layers.Concatenate(name="hand_board_concat")(inputs)
        return self.value_head(stacked_emb)

    def get_config(self):
        return super().get_config()


def build_value_model(config):
    """
    Builds complete model for hand_value training mode:
        - input grid with two channels for hand/board
        - combined grid created from hand/board grids
        - CardStateEncoder created embeddings for hand/board/combined
        - CombinedInputValueHead creates feed forward network from combined embedding input
        - SeparateInputValueHead creates feed forward network from hand and board embedding inputs
        - value head has sigmoid output representing hand strength from 0 (strongest) to 1 (weakest)

    Args:
        - config (dict): config for models
    """
    inputs = tf.keras.Input(config["input_shape"], name="poker_input") # (batch, 13, 4, 2)
    
    # --- Split and combine grids ---
    hand = inputs[..., 0:1]         # (batch, 13, 4, 1)
    board = inputs[..., 1:2]        # (batch, 13, 4, 1)
    combo = hand + board

    encoder_config = get_encoder_config(config)
    encoder = CardStateEncoder(encoder_config)
    hand_emb, board_emb, combo_emb = encoder([hand, board, combo])

    submode = config.get("submode", "combined")
    if submode == "combined":
        value_head = CombinedInputValueHead()
        outputs = value_head(combo_emb, training=True)
    else:
        value_head = SeparateInputValueHead()
        outputs = value_head([hand_emb, board_emb], training=True)

    # value_heads = CardStateGridValueHead()
    # hand_value, board_value, combined_value = value_heads([hand_emb, board_emb, combo_emb], return_all=True)

    # --- Build Model ---
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # model = tf.keras.Model(inputs=inputs, outputs=[hand_value, board_value, combined_value])
    
    return model