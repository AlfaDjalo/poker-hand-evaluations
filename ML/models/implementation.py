import itertools
import tensorflow as tf
from tensorflow.keras import layers, Model

class PokerCNNEncoder(Model):
    def __init__(self, input_shape, filters=(8, 16, 32), kernel_size=2, embedding_dim=32, use_equivariance=True):
        super().__init__()
        self.convs = [
            layers.Conv2D(f, kernel_size, padding="same", activation="relu")
            for f in filters
        ]
        self.bn = [layers.BatchNormalization() for _ in filters]
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation="relu")
        self.dense2 = layers.Dense(64, activation="relu")
        self.dropout = layers.Dropout(0.2)
        self.embedding = layers.Dense(embedding_dim, activation=None)  # <-- embeddings live here
        if use_equivariance == True:
            self.equiv = SuitEquivariantLayer(pooling="mean")
        else:
            self.equiv = None

        # Build model by calling once (if using subclassing)
        # self.build((None, *input_shape))

    def call(self, inputs, training=False):
        if self.equiv is not None:
            x = self.equiv(inputs)
        else:
            x = inputs
        # x = SuitEquivariantLayer(pooling="mean")(x)
        # x = tf.expand_dims(x, axis=-1) # add channel dim for Conv2D
        # x = layers.Conv2D(8, (2, 2), padding="same", activation="relu")(x)
        for conv, bn in zip(self.convs, self.bn):
            x = conv(x)
            x = bn(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.embedding(x)
        return x  # embedding vector


class PokerValueModel(Model):
    def __init__(self, encoder, activation="sigmoid"):
        super().__init__()
        self.encoder = encoder
        self.value_head = layers.Dense(1, activation=activation)

    def call(self, inputs, training=False):
        embedding = self.encoder(inputs, training=training)
        value = self.value_head(embedding)
        return value


class PokerValueHeads(tf.keras.Model):
    def __init__(self, encoders, activation="sigmoid"):
        super().__init__()
        self.hand_value = PokerValueModel(encoders.hand_encoder, activation)
        self.board_value = PokerValueModel(encoders.board_encoder, activation)
        self.combined_value = PokerValueModel(encoders.combined_encoder, activation)
        
    def call(self, inputs, training=False, return_all=False):
        hand, board, combo = inputs

        hand_v = self.hand_value(hand, training=training)
        board_v = self.board_value(board, training=training)
        combined_v = self.combined_value(combo, training=training)

        if return_all:
            return hand_v, board_v, combined_v
        
        return combined_v

class PokerValueHeads(tf.keras.Model):
    def __init__(self, encoders, activation="sigmoid"):
        super().__init__()
        self.hand_value = PokerValueModel(encoders.hand_encoder, activation)
        self.board_value = PokerValueModel(encoders.board_encoder, activation)
        self.combined_value = PokerValueModel(encoders.combined_encoder, activation)

    def call(self, inputs, training=False, return_all=False):
        hand, board, combo = inputs
        # combo = tf.concat([hand, board, hand + board], axis=-1)

        hand_v = self.hand_value(hand, training=training)
        board_v = self.board_value(board, training=training)
        combined_v = self.combined_value(combo, training=training)

        if return_all:
            return hand_v, board_v, combined_v
        return combined_v



class SuitEquivariantLayer(tf.keras.layers.Layer):
    def __init__(self, pooling="mean", **kwargs):
        super().__init__(**kwargs)
        self.pooling = pooling
        self.P = get_permutation_matrices(4)    # shape (24, 4, 4)

    def call(self, inputs):
        # inputs: (batch, 13, 4)
        # Expand for broadcast
        # x = tf.expand_dims(inputs, axis=1)      # (batch, 1, 13, 4)
        P = tf.cast(self.P, inputs.dtype)            # (24, 4, 4)

        # Apply all 24 permutations to the suit axis
        # Equivalent to: for each P_i, X @ P_i
        X_perm = tf.einsum("brsf, psm->bprmf", inputs, P)  # (batch, 24, 13, 4)

        # Pool across permutations to enforce invariance
        if self.pooling == "mean":
            X_pooled = tf.reduce_mean(X_perm, axis=1)   # (batch, 13, 4)
        elif self.pooling == "max":
            X_pooled = tf.reduce_max(X_perm, axis=1)
        else:
            raise ValueError("Unsupported pooling type")
        
        return X_pooled
        

class PokerComboModel(tf.keras.Model):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.hand_encoder = PokerCNNEncoder(input_shape=(13, 4, 1), embedding_dim=embedding_dim, use_equivariance=True)
        self.board_encoder = PokerCNNEncoder(input_shape=(13, 4, 1), embedding_dim=embedding_dim, use_equivariance=True)
        self.combined_encoder = PokerCNNEncoder(input_shape=(13, 4, 1), embedding_dim=embedding_dim, use_equivariance=True)

    def call(self, inputs, training=False, return_all=True):
        # inputs: (batch, 13, 4, 2)
        hand = inputs[..., 0:1] # (batch, 13, 4, 1)
        board = inputs[..., 1:2] # (batch, 13, 4, 1)

        # Compute a 'combined' grid 
        combined = hand + board

        # Stack the three grids along the channel axis
        combo_input = tf.concat([hand, board, combined], axis=-1) # (batch, 13, 4, 3)

        # Feed to encoder
        hand_emb = self.hand_encoder(hand, training=training)
        board_emb = self.board_encoder(board, training=training)
        combined_emb = self.combined_encoder(combo_input, training=training)

        if return_all:
            return hand_emb, board_emb, combined_emb

        return combined_emb
    




def get_permutation_matrices(n=4):
    perms = list(itertools.permutations(range(n)))
    matrices = []
    for p in perms:
        mat = tf.one_hot(p, depth=n)
        matrices.append(mat)
    return tf.constant(tf.stack(matrices), dtype=tf.float32)

def main():
    mat = get_permutation_matrices(4)
    print(mat)

if __name__ == "__main__":
    main()