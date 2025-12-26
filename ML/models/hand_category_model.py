import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable
from .encoders import *

@register_keras_serializable(package="Poker")
class HandCategoryHead(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category_head = layers.Dense(9, activation="softmax")

    def call(self, inputs, training=False):
        return self.category_head(inputs)

@register_keras_serializable(package="Poker")
class CardStateHandCategoryHead(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hand_category = HandCategoryHead()
        self.board_category = HandCategoryHead()
        self.combined_category = HandCategoryHead()

    def call(self, inputs, training=False, return_all=True):
        hand, board, combo = inputs

        hand_cat = self.hand_category(hand, training=training)
        board_cat = self.board_category(board, training=training)
        combined_cat = self.combined_category(combo, training=training)

        if return_all:
            return hand_cat, board_cat, combined_cat
        
        return combined_cat

def build_hand_category_model(config):
    """
    Builds complete model for hand_category training mode:
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

    encoder_config = get_encoder_config(config)
    encoder = CardStateEncoder(encoder_config)
    hand_emb, board_emb, combo_emb = encoder([hand, board, combo])

    category_heads = CardStateHandCategoryHead()
    hand_cat, board_cat, combined_cat = category_heads([hand_emb, board_emb, combo_emb], return_all=True)

    # --- Build Model ---
    model = tf.keras.Model(inputs=inputs, outputs=[hand_cat, board_cat, combined_cat])

    return model

@register_keras_serializable(package="Poker")
class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_weights, from_logits=False, name="weighted_cce"):
        super().__init__(name=name)
        self.class_weights = tf.reshape(class_weights, (1, -1))
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        weights = tf.reduce_sum(self.class_weights * y_true, axis=-1)
        return ce * weights
