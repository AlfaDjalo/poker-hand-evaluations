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
    # def __init__(self, hand_encoder, board_encoder, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        # self.hand_encoder = hand_encoder
        # self.board_encoder = hand_encoder
        self.value_head = layers.Dense(1, activation="sigmoid")
        # self._value_config = {"activation": activation, "encoder_class": encoder.__class__.__name__}

        # Projection layers
        # self.d1 = layers.Dense(128, activation="relu")
        # self.d2 = layers.Dense(64, activation="relu")
        # self.out = layers.Dense(1, activation=activation)

    def call(self, inputs, training=False):

        # --- Embedding composition ---

        stacked_emb = tf.keras.layers.Concatenate(name="hand_board_concat")(inputs)
        # stacked_emb = tf.keras.layers.Concatenate(name="hand_board_concat")(
        #     [hand_emb, board_emb]
        # )

        # hand_embedding = self.hand_encoder(inputs, training=training)
        # embedding = self.hand_encoder(inputs, training=training)
        return self.value_head(stacked_emb)

    def get_config(self):
        return super().get_config()
        # base = super().get_config()
        # base.update({"config": dict(self._value_config)})
        # return base

    # self._config = {
    #         "activation": activation,
    #         "encoder_class": encoder.__class__.__name__,
    #     }

    # def call(self, inputs, training=False):
    #     hand, board = inputs

    #     hand_emb = self.encoder(hand, training=training)
    #     board_emb = self.encoder(board, training=training)

    #     # Compose embeddings explicitly
    #     x = tf.concat(
    #         [hand_emb, board_emb, hand_emb + board_emb],
    #         axis=-1
    #     )

    #     x = self.d1(x)
    #     x = self.d2(x)
    #     return self.out(x)

    # def get_config(self):
    #     base = super().get_config()
    #     base.update({"config": dict(self._config)})
    #     return base



# @register_keras_serializable(package="Poker")
# class CardStateEmbeddingValueHead(tf.keras.Model):
#     """
#     Value heads operating on (hand_emb, board_emb).
#     """
#     def __init__(self, encoders, activation="sigmoid", **kwargs):
#         super().__init__(**kwargs)
#         self.encoders = encoders

#         self.hand_board_value = EmbeddingValueHead(
#             self.encoders.hand_encoder,
#             activation=activation
#         )

#         self.activation = activation

#     def call(self, inputs, training=False):
#         hand, board = inputs
#         return self.hand_board_value([hand, board], training=training)

#     def get_config(self):
#         base = super().get_config()
#         base.update({
#             "activation": self.activation,
#             "encoders_config": self.encoders.get_config(),
#         })
#         return base


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
    # combo = ComboConcatLayer(name="combo_concat")([hand, board])

    encoder_config = get_encoder_config(config)
    encoder = CardStateEncoder(encoder_config)
    hand_emb, board_emb, combo_emb = encoder([hand, board, combo])

    # hand_emb = encoder(hand, training=True)
    # board_emb = encoder(board, training=True)

    # # --- Embedding composition ---
    # stacked_emb = tf.keras.layers.Concatenate(name="hand_board_concat")(
    #     [hand_emb, board_emb]
    # )

    value_head = EmbeddingValueHead()
    value = value_head([hand_emb, board_emb], training=True)


    # # --- Small value head (embedding-only) ---
    # for i, units in enumerate(config["embedding_value_head_units"]):
    #     x = tf.keras.layers.Dense(
    #         units,
    #         activation=config["activation"],
    #         name=f"embedding_value_dense_{i}"
    #     )(x)

    # value = tf.keras.layers.Dense(
    #     1,
    #     activation="linear",
    #     name="embedding_value"
    # )(x)

    model = tf.keras.Model(inputs=inputs, outputs=value)

    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
    #     loss="mse",
    #     metrics=["mae"]
    # )

    return model
