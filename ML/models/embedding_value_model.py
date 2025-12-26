import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable
from .encoders import *

@register_keras_serializable(package="Poker")
class EmbeddingValueHead(Model):
    """
    Value head that operates directly on hand and board embeddings,
    instead of a combined grid.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value_head = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        stacked_emb = tf.keras.layers.Concatenate(name="hand_board_concat")(inputs)
        return self.value_head(stacked_emb)

    def get_config(self):
        return super().get_config()


def build_embedding_value_model(config):
    """
    Builds a model that predicts value using ONLY hand and board embeddings.

    Purpose:
        - Enforce composability of embeddings
        - Mimic RL-style consumption of embeddings
        - Provide auxiliary training signal for the encoder

    Architecture:
        grid -> encoder -> embeddings
        embeddings -> small MLP -> scalar value
    """

    # --- Inputs ---
    inputs = tf.keras.Input(config["input_shape"], name="poker_input") # (batch, 13, 4, 2)

    # --- Split and combine grids ---
    hand = inputs[..., 0:1]         # (batch, 13, 4, 1)
    board = inputs[..., 1:2]        # (batch, 13, 4, 1)
    combo = hand + board

    encoder_config = get_encoder_config(config)
    encoder = CardStateEncoder(encoder_config)
    hand_emb, board_emb, combo_emb = encoder([hand, board, combo])

    value_head = EmbeddingValueHead()
    value = value_head([hand_emb, board_emb], training=True)

    model = tf.keras.Model(inputs=inputs, outputs=value)

    return model
