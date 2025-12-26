import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable
from .encoders import *

@register_keras_serializable(package="Poker")
class GridValueHead(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value_head = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        # embedding = self.encoder(inputs, training=training)
        return self.value_head(inputs)
        # return self.value_head(embedding)

    # def get_config(self):
    #     base = super().get_config()
    #     base.update({"config": dict(self._value_config)})
    #     return base

    # def from_config(cls, config):
    #     cfg = config.pop("config", {})
    #     return cls(encoder=None, **cfg)

    # @property
    # def encoder_input_shape(self, config):
    #     # Match the encoder’s expected shape from config
    #     return config["encoder_input_shape"] #(14, 4, 2)

@register_keras_serializable(package="Poker")
class CardStateGridValueHead(tf.keras.Model):
    """
    Model to create value models for hand, board and combined.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # if encoders is None:
        #     raise ValueError("CardStateGridValueHead requires encoders parameter at construction time.")
        # self.encoders = encoders
        self.hand_value = GridValueHead()
        self.board_value = GridValueHead()
        self.combined_value = GridValueHead()

    def call(self, inputs, training=False, return_all=True):
        hand, board, combo = inputs

        hand_value = self.hand_value(hand, training=training)
        board_value = self.board_value(board, training=training)
        combined_value = self.combined_value(combo, training=training)

        if return_all:
            return hand_value, board_value, combined_value
        
        return combined_value

    # def get_config(self):
    #     base = super().get_config()
    #     base.update({
    #         "encoders_config": self.encoders.get_config(),
    #     })
    #     return base

    # @classmethod
    # def from_config(cls, config):
    #     enc_cfg = config.pop("encoders_config", None)
    #     if enc_cfg is None:
    #         raise ValueError("encoders_config is required to reconstruct CardStateGridValueHeads")
    #     # PokerComboModel.from_config expects the dict we returned in get_config
    #     encoders = CardStateEncoder.from_config(enc_cfg)
    #     return cls(encoders=encoders, **config)


def build_grid_value_model(config):
    """
    Builds complete model for absolute_value training mode:
        - input grid with three channels for hand/board/combined
        - PokerComboModel contains three encoders, one each for hand/board/combined
        - PokerValueHead contains three feed forward networks, one for each of hand/board/combined
        - three sigmoid outputs, one for each of hand/value/combined

    Args:
        - config (dict): config for models
    """
    inputs = tf.keras.Input(config["input_shape"], name="poker_input") # (batch, 13, 4, 2)
    
    # --- Split and combine grids ---
    hand = inputs[..., 0:1]         # (batch, 13, 4, 1)
    board = inputs[..., 1:2]        # (batch, 13, 4, 1)
    combo = hand + board
    # combo = ComboConcatLayer(name="combo_concat")([hand, board])

    encoder_config = get_encoder_config(config)
    encoder = CardStateEncoder(encoder_config)
    hand_emb, board_emb, combo_emb = encoder([hand, board, combo])

    value_heads = CardStateGridValueHead()
    # value_heads = CardStateGridValueHead(encoder)
    hand_value, board_value, combined_value = value_heads([hand_emb, board_emb, combo_emb], return_all=True)

    # --- Build Model ---
    model = tf.keras.Model(inputs=inputs, outputs=[hand_value, board_value, combined_value])
    
    # # --- Compile Model ---
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
    #     loss=["mse", "mse", "mse"],
    #     loss_weights=config["loss_weights"],
    #     metrics=["mae", "mae", "mae"]
    # )

    return model