import tensorflow as tf
from keras.saving import register_keras_serializable
from .encoders import PokerCNNEncoder

@register_keras_serializable(package="Poker")
class ComboConcatLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        hand, board = inputs
        return hand + board

    def compute_output_shape(self, input_shape):
        batch_size, h, w, c = input_shape[0]
        return (batch_size, h, w, c)

@register_keras_serializable(package="Poker")
class PokerComboModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self._encoder_config = dict(get_encoder_config(config)) if isinstance(config, dict) else dict(config)
        self.use_shared_encoder = config.get("use_shared_encoder", False)
        if self.use_shared_encoder:
            self.shared_encoder = PokerCNNEncoder(self._encoder_config)
            self.hand_encoder = self.shared_encoder
            self.board_encoder = self.shared_encoder
            self.combined_encoder = self.shared_encoder
        else:
            self.hand_encoder = PokerCNNEncoder(self._encoder_config)
            self.board_encoder = PokerCNNEncoder(self._encoder_config)
            self.combined_encoder = PokerCNNEncoder(self._encoder_config)

    def call(self, inputs, training=False, return_all=True):
        # robustly handle single- or two-channel inputs
        ch = inputs.shape[-1]
        if ch == 2:
            hand = inputs[..., 0:1]; board = inputs[..., 1:2]
        else:
            hand = inputs[..., 0:1]; board = tf.zeros_like(hand)
        combined = hand + board
        hand_emb = self.hand_encoder(hand, training=training)
        board_emb = self.board_encoder(board, training=training)
        combined_emb = self.combined_encoder(combined, training=training)
        if return_all:
            return hand_emb, board_emb, combined_emb
        return combined_emb

def get_encoder_config(global_config):
    return {
        "input_shape_encoder": global_config["input_shape_encoder"],
        "filters": global_config["filters"],
        "kernel_size": global_config["kernel_size"],
        "embedding_dim": global_config["embedding_dim"],
        "use_equivariance": global_config["use_equivariance"],
        "use_shared_encoder": global_config.get("use_shared_encoder", False),
    }
