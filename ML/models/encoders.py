import itertools
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable

# Encoder Model
@register_keras_serializable(package="Poker")
class CardSetEncoder(Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        self.encoder_input_shape = config["encoder_input_shape"]  # ideally (14,4,C)
        self.embedding_dim = config["embedding_dim"]

        # First layer: full suit mixing + local rank mixing
        self.conv2d = layers.Conv2D(
            filters=32,
            kernel_size=(2,4),
            padding="valid",
            activation="relu"
        )
        self.bn2d = layers.BatchNormalization()
        
        # After conv2d, reshape to (batch, ranks-1, filters)
        self.reshape = layers.Reshape((13, 32))
        # self.reshape = layers.Reshape((-1, 32))
        
        # Rank-progressive 1D layers
        self.conv1 = layers.Conv1D(48, kernel_size=2, padding="valid", activation="relu")
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv1D(64, kernel_size=2, padding="valid", activation="relu")
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv1D(96, kernel_size=2, padding="valid", activation="relu")
        self.bn3 = layers.BatchNormalization()

        # Dense projection to embedding
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(256, activation="relu")
        self.d2 = layers.Dense(64, activation="relu")
        self.embedding = layers.Dense(self.embedding_dim, activation=None)

        # L2 normalisation
        self.l2_norm = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))

    def call(self, x, training=False):

        x = self.conv2d(x)
        x = self.bn2d(x, training=training)

        x = self.reshape(x)

        x = self.conv1(x)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)

        x = self.embedding(x)
        x = self.l2_norm(x)

        return x

    def get_config(self):
        base = super().get_config()
        base.update({
            "config": {
                "input_shape_encoder": self.encoder_input_shape,
                # "filters": self.filters,
                # "kernel_size": self.kernel_size,
                "embedding_dim": self.embedding_dim,
                # "use_equivariance": self.use_equivariance,
            }
        })
        return base

    @classmethod
    def from_config(cls, config):
        if "config" in config:
            cfg = config.pop("config")
        else:
            cfg = config
        return cls(cfg, **config)


# @register_keras_serializable(package="Poker")
# class SuitPermutationLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.perms = list(itertools.permutations(range(4)))

#     def call(self, inputs):
#         # inputs: (batch, 13, 4, channels)
#         permuted = []
#         for perm in self.perms:
#             permuted.append(tf.gather(inputs, indices=list(perm), axis=2))
#         # Stack on new axis for permutations
#         return tf.stack(permuted, axis=1) # shape: (batch, 24, 13, 4, channels)
    
#     def get_config(self):
#         return super().get_config()


# @register_keras_serializable(package="Poker")
# class ComboConcatLayer(tf.keras.layers.Layer):
#     def call(self, inputs):
#         hand, board = inputs
#         return hand + board

#     def compute_output_shape(self, input_shape):
#         batch_size, h, w, c = input_shape[0]
#         return (batch_size, h, w, c)


@register_keras_serializable(package="Poker")
class CardStateEncoder(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self._encoder_config = dict(get_encoder_config(config)) if isinstance(config, dict) else dict(config)
        self.card_set_encoder = CardSetEncoder(self._encoder_config)
        # self.use_shared_encoder = config.get("use_shared_encoder", False)
        # if self.use_shared_encoder:
        #     self.shared_encoder = CardSetEncoder(self._encoder_config)
        #     self.hand_encoder = self.shared_encoder
        #     self.board_encoder = self.shared_encoder
        #     self.combined_encoder = self.shared_encoder
        # else:
        #     self.hand_encoder = CardSetEncoder(self._encoder_config)
        #     self.board_encoder = CardSetEncoder(self._encoder_config)
        #     self.combined_encoder = CardSetEncoder(self._encoder_config)

    def call(self, inputs, training=False):
    # def call(self, inputs, training=False, return_all=True):
        # robustly handle single- or two-channel inputs
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return [self.card_set_encoder(x, training=training) for x in inputs]        
           
        # ch = inputs.shape[-1]
        # if ch == 2:
        #     hand = inputs[..., 0:1]; board = inputs[..., 1:2]
        # else:
        #     hand = inputs[..., 0:1]; board = tf.zeros_like(hand)
        # combined = hand + board
        # hand_emb = self.hand_encoder(hand, training=training)
        # board_emb = self.board_encoder(board, training=training)
        # combined_emb = self.combined_encoder(combined, training=training)
        # if return_all:
        #     return hand_emb, board_emb, combined_emb
        # return combined_emb

    def get_config(self):
        base = super().get_config()
        base.update({
            "encoder_config": self._encoder_config,
        })
        return base

    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop("encoder_config")
        return cls(encoder_config, **config)


def get_encoder_config(global_config):
    return {
        "encoder_input_shape": global_config["encoder_input_shape"],
        "filters": global_config["filters"],
        "kernel_size": global_config["kernel_size"],
        "embedding_dim": global_config["embedding_dim"],
        "use_equivariance": global_config["use_equivariance"],
        "use_shared_encoder": global_config.get("use_shared_encoder", False),
    }
