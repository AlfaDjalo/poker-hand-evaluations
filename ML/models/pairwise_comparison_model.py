import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable
from .encoders import *


@register_keras_serializable(package="Poker")
class PairwiseComparisonHead(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.encoder = encoder
        self.pairwise_head = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        embedding_A, embedding_B = inputs
        # embedding_A = self.encoder(input_A, training=training)
        # embedding_B = self.encoder(input_B, training=training)
        combined_embeddings = tf.concat([embedding_A, embedding_B], axis=-1)
        return self.pairwise_head(combined_embeddings)

@register_keras_serializable(package="Poker")
class CardStatePairwiseComparisonHead(tf.keras.Model):
    def __init__(self, **kwargs):
    # def __init__(self, encoders=None, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        # if encoders is None:
        #     raise ValueError("PairwiseComparisonHeads requires encoders parameter at construction time.")
        # self.encoders = encoders
        self.hand_comparison = PairwiseComparisonHead()
        self.board_comparison = PairwiseComparisonHead()
        self.combined_comparison = PairwiseComparisonHead()
        # self.activation = activation

    def call(self, inputs, training=False, return_all=True):
        x1, x2 = inputs
        hand1, board1, combo1 = x1
        hand2, board2, combo2 = x2
        hand_prob = self.hand_comparison([hand1, hand2], training=training)
        board_prob = self.board_comparison([board1, board2], training=training)
        combined_prob = self.combined_comparison([combo1, combo2], training=training)
        if return_all:
            return hand_prob, board_prob, combined_prob
        return combined_prob

def build_pairwise_comparison_model(config):
    # inputs = tf.keras.Input(config["input_shape"], name="poker_input")
    
    # --- Split and combine grids ---
    inputs_A = tf.keras.Input(config["input_shape"], name="input_A")
    inputs_B = tf.keras.Input(config["input_shape"], name="input_B")
    
    hand_A = inputs_A[..., 0:1]         # (batch, 13, 4, 1)
    board_A = inputs_A[..., 1:2]        # (batch, 13, 4, 1)
    combo_A = hand_A + board_A
    # combo_A = ComboConcatLayer(name="combo_concat_A")([hand_A, board_A])

    hand_B = inputs_B[..., 0:1]         # (batch, 13, 4, 1)
    board_B = inputs_B[..., 1:2]        # (batch, 13, 4, 1)
    combo_B = hand_B + board_B
    # combo_B = ComboConcatLayer(name="combo_concat_B")([hand_B, board_B])

    encoder_config = get_encoder_config(config)
    encoder = CardStateEncoder(encoder_config)
    hand_emb_A, board_emb_A, combo_emb_A = encoder([hand_A, board_A, combo_A])
    hand_emb_B, board_emb_B, combo_emb_B = encoder([hand_B, board_B, combo_B])

    pairwise_comparison_heads = CardStatePairwiseComparisonHead()
    hand_prob, board_prob, combined_prob = pairwise_comparison_heads([(hand_emb_A, board_emb_A, combo_emb_A), (hand_emb_B, board_emb_B, combo_emb_B)], training=True, return_all=True)
    # hand_value, board_value, combined_value = value_heads([hand_emb, board_emb, combo_emb], return_all=True)

    # --- Build Model ---
    model = tf.keras.Model(inputs=[inputs_A, inputs_B], outputs=[hand_prob, board_prob, combined_prob])
    
    # # --- Compile Model ---
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
    #     loss=["mse", "mse", "mse"],
    #     loss_weights=config["loss_weights"],
    #     # loss_weights=[0.3, 0.3, 0.4],
    #     metrics=["mae", "mae", "mae"]
    # )

    return model