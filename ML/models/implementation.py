import itertools
import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable
from typing import Optional

@register_keras_serializable(package="Poker")
class PokerCNNEncoder(Model):
    """
    Model to convert input grid into embedding vector.
    
    Inputs:
        input_shape_encoder: 13 x 4 grid

    Outputs:
        embedding: embedding_dim = 32d vector
    """
    def __init__(self, config, **kwargs):
        """
        config: dict with keys:
          - input_shape_encoder (tuple)
          - filters (tuple)
          - kernel_size (int)
          - embedding_dim (int)
          - use_equivariance (bool)
        """        
        super().__init__(**kwargs)
        self._init_config = dict(config)
        self.input_shape_encoder = config["input_shape_encoder"]
        print(f"🔍 Creating encoder with input_shape: {self.input_shape_encoder}")  # ADD THIS
        self.filters = config["filters"]
        self.kernel_size = config["kernel_size"]
        self.embedding_dim = config["embedding_dim"]
        self.use_equivariance = config["use_equivariance"]
        
    # def __init__(self, input_shape, filters=(8, 16, 32), kernel_size=2, embedding_dim=32, use_equivariance=True, **kwargs):
    #     super().__init__(**kwargs)
    #     self.input_shape = input_shape
    #     self.filters = filters
    #     self.kernel_size = kernel_size
    #     self.embedding_dim = embedding_dim
    #     self.use_equivariance = use_equivariance

        self.convs = [
            layers.Conv2D(f, self.kernel_size, padding="same", activation="relu")
            for f in self.filters
        ]
        self.bn = [layers.BatchNormalization() for _ in self.filters]
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation="relu")
        self.dense2 = layers.Dense(64, activation="relu")
        self.dropout = layers.Dropout(0.2)
        self.embedding = layers.Dense(self.embedding_dim, activation=None)  # <-- embeddings live here
        self.equiv = SuitEquivariantLayer(pooling="mean")

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
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.embedding(x)
        return x  # embedding vector

    def get_config(self):
        base = super().get_config()
        base.update({
            "config": {
                "input_shape_encoder": self.input_shape_encoder,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "embedding_dim": self.embedding_dim,
                "use_equivariance": self.use_equivariance,
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


@register_keras_serializable(package="Poker")
class PokerValueModel(Model):
    """
    Model to convert embedding into hand rank value
    
    Inputs:
        encoder model

    Outputs:
        hand rank value: ***integer*** (should be, not yet) in range(1, 7460 ?)
    """
    def __init__(self, encoder, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.value_head = layers.Dense(1, activation=activation)
        self._value_config = {"activation": activation, "encoder_class": encoder.__class__.__name__}

    def call(self, inputs, training=False):
        embedding = self.encoder(inputs, training=training)
        value = self.value_head(embedding)
        return value

    def get_config(self):
        base = super().get_config()
        base.update({"config": dict(self._value_config)})
        # base.update({
        #     "activation": self.value_head.activation.__name__,
        #     "encoder_class": self.encoder.__class__.__name__,
        # })
        return base

    def from_config(cls, config):
        cfg = config.pop("config", {})
        return cls(encoder=None, activation=cfg.get("activation", "sigmoid"), **config)

    @property
    def encoder_input_shape(self):
        # Match the encoder’s expected shape from config
        return (13, 4, 2)

@register_keras_serializable(package="Poker")
class PokerValueHeads(tf.keras.Model):
    """
    Model to create encoder/value models for hand, board and combined.
    """
    def __init__(self, encoders=None, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        if encoders == None:
            raise ValueError("PokerValueHeads requires encoders parameter at construction time.")
        self.encoders = encoders
        self.hand_value = PokerValueModel(self.encoders.hand_encoder, activation)
        self.board_value = PokerValueModel(self.encoders.board_encoder, activation)
        self.combined_value = PokerValueModel(self.encoders.combined_encoder, activation)
        self.activation = activation

    def call(self, inputs, training=False, return_all=True):
    # def call(self, inputs, training=False, return_all=False):
        hand, board, combo = inputs
        print(f"🔍 Combo shape received: {combo.shape}")

        hand_v = self.hand_value(hand, training=training)
        board_v = self.board_value(board, training=training)
        combined_v = self.combined_value(combo, training=training)

        if return_all:
            return hand_v, board_v, combined_v
        
        return combined_v

    def get_config(self):
        base = super().get_config()
        base.update({
            "activation": self.activation,
            "encoders_config": self.encoders.get_config(),
        })
        return base

    @classmethod
    def from_config(cls, config):
        activation = config.pop("activation", "sigmoid")
        enc_cfg = config.pop("encoders_config", None)
        if enc_cfg is None:
            raise ValueError("encoders_config is required to reconstruct PokerValueHeads")
        # PokerComboModel.from_config expects the dict we returned in get_config
        encoders = PokerComboModel.from_config(enc_cfg)
        return cls(encoders=encoders, activation=activation, **config)

@register_keras_serializable(package="Poker")
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
        
    def get_config(self):
        base = super().get_config()
        base.update({ "pooling": self.pooling })
        return base
    
    @ classmethod
    def from_config(cls, config):
        return cls(**config)



@register_keras_serializable(package="Poker")
class PokerComboModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
    # def __init__(self, embedding_dim=32, **kwargs):
        super().__init__(**kwargs)
        self._encoder_config = dict(get_encoder_config(config)) if isinstance(config, dict) else dict(config)
        self.hand_encoder = PokerCNNEncoder(self._encoder_config)
        self.board_encoder = PokerCNNEncoder(self._encoder_config)
        self.combined_encoder = PokerCNNEncoder(self._encoder_config)
        print(f"🔍 PokerComboModel encoder config: {self._encoder_config}")  # ADD THIS

        # self.hand_encoder = PokerCNNEncoder(input_shape=(13, 4, 1), embedding_dim=embedding_dim, use_equivariance=True)
        # self.board_encoder = PokerCNNEncoder(input_shape=(13, 4, 1), embedding_dim=embedding_dim, use_equivariance=True)
        # self.combined_encoder = PokerCNNEncoder(input_shape=(13, 4, 1), embedding_dim=embedding_dim, use_equivariance=True)

    def call(self, inputs, training=False, return_all=True):
        # inputs: (batch, 13, 4, 2)
        hand = inputs[..., 0:1] # (batch, 13, 4, 1)
        board = inputs[..., 1:2] # (batch, 13, 4, 1)

        # Compute a 'combined' grid 
        # combined = hand + board

        # Stack the three grids along the channel axis
        # combo_input = tf.concat([hand, board, combined], axis=-1) # (batch, 13, 4, 3)
        combo_input  = hand + board

        # Feed to encoder
        hand_emb = self.hand_encoder(hand, training=training)
        board_emb = self.board_encoder(board, training=training)
        combined_emb = self.combined_encoder(combo_input, training=training)

        if return_all:
            return hand_emb, board_emb, combined_emb

        return combined_emb

    def get_config(self):
        base = super().get_config()
        base.update({
            "config": self._encoder_config
            # {
                # "input_shape_encoder": self.hand_encoder.input_shape,
                # "filters": self.hand_encoder.filters,
                # "kernel_size": self.hand_encoder.kernel_size,
                # "embedding_dim": self.hand_encoder.embedding_dim,
                # "use_equivariance": self.hand_encoder.use_equivariance,
            # }
        })
        return base

    @classmethod
    def from_config(cls, config):
        cfg = config.pop("config")
        return cls(cfg, **config)

@register_keras_serializable(package="Poker")
class ComboConcatLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        hand, board = inputs
        return hand+board        
        # return tf.concat([hand, board, hand+board], axis=-1)        

    def compute_output_shape(self, input_shape):
        batch_size, h, w, c = input_shape[0]
        return (batch_size, h, w, c)
        # return (batch_size, h, w, c*3)

def build_value_model(config): # build_value_model ?
    inputs = tf.keras.Input(config["input_shape"], name="poker_input")
    # inputs = tf.keras.Input(shape=(13, 4, 2), name="poker_input")
    
    # --- Split and combine grids ---
    hand = inputs[..., 0:1]         # (batch, 13, 4, 1)
    board = inputs[..., 1:2]        # (batch, 13, 4, 1)
    combo = ComboConcatLayer(name="combo_concat")([hand, board])
    # combo = Lambda(lambda x:ops.concatenate([x[0], x[1], x[0]+x[1]], axis=-1), output_shape=(13, 4, 3))([hand, board])

    encoder_cfg = get_encoder_config(config)
    encoder = PokerComboModel(encoder_cfg)
    # encoder = PokerComboModel(embedding_dim=config["embedding_dim"])
    value_heads = PokerValueHeads(encoder, activation=config["activation"])
    hand_v, board_v, combined_v = value_heads([hand, board, combo], training=True, return_all=True)

    # --- Build Model ---
    model = tf.keras.Model(inputs=inputs, outputs=[hand_v, board_v, combined_v])
    
    # --- Compile Model ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        loss=["mse", "mse", "mse"],
        loss_weights=config["loss_weights"],
        # loss_weights=[0.3, 0.3, 0.4],
        metrics=["mae", "mae", "mae"]
    )

    return model

@register_keras_serializable(package="Poker")
class PairwiseComparisonModel(Model):
    """
    Takes embeddings for A and B, and outputs sigmoid probability that A > B.

    Inputs:
        encoder model

    Outputs:
        probaility input A > input B
    """
    def __init__(self, encoder, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.pairwise_head = layers.Dense(1, activation=activation)
        self._pairwise_config = {"activation": activation, "encoder_class": encoder.__class__.__name__}

    def call(self, inputs, training=False):
        input_A, input_B = inputs

        embedding_A = self.encoder(input_A, training=training)
        embedding_B = self.encoder(input_B, training=training)
        combined_embeddings = tf.concat([embedding_A, embedding_B], axis=-1)
        
        prob = self.pairwise_head(combined_embeddings)
        return prob

    def get_config(self):
        base = super().get_config()
        base.update({"config": dict(self._pairwise_config)})
        # base.update({
        #     "activation": self.value_head.activation.__name__,
        #     "encoder_class": self.encoder.__class__.__name__,
        # })
        return base

    def from_config(cls, config):
        cfg = config.pop("config", {})
        return cls(encoder=None, activation=cfg.get("activation", "sigmoid"), **config)

    @property
    def encoder_input_shape(self):
        # Match the encoder’s expected shape from config
        return self.encoder.input_shape_encoder


@register_keras_serializable(package="Poker")
class PairwiseComparisonHeads(tf.keras.Model):
    """
    Model to create encoder/pairwise comparison models for hand, board and combined.
    """
    def __init__(self, encoders=None, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        if encoders == None:
            raise ValueError("PairwiseComparisonHeads requires encoders parameter at construction time.")
        self.encoders = encoders
        self.hand_comparison = PairwiseComparisonModel(self.encoders.hand_encoder, activation)
        self.board_comparison = PairwiseComparisonModel(self.encoders.board_encoder, activation)
        self.combined_comparison = PairwiseComparisonModel(self.encoders.combined_encoder, activation)
        self.activation = activation

    def call(self, inputs, training=False, return_all=True):
    # def call(self, inputs, training=False, return_all=False):
  
        x1, x2 = inputs

        hand1, board1, combo1 = x1        
        hand2, board2, combo2 = x2        

        hand_prob = self.hand_comparison([hand1, hand2], training=training)
        board_prob = self.board_comparison([board1, board2], training=training)
        combined_prob = self.combined_comparison([combo1, combo2], training=training)

        if return_all:
            return hand_prob, board_prob, combined_prob
        
        return combined_prob

    def get_config(self):
        base = super().get_config()
        base.update({
            "activation": self.activation,
            "encoders_config": self.encoders.get_config(),
        })
        return base

    @classmethod
    def from_config(cls, config):
        activation = config.pop("activation", "sigmoid")
        enc_cfg = config.pop("encoders_config", None)
        if enc_cfg is None:
            raise ValueError("encoders_config is required to reconstruct PokerValueHeads")
        # PokerComboModel.from_config expects the dict we returned in get_config
        encoders = PokerComboModel.from_config(enc_cfg)
        return cls(encoders=encoders, activation=activation, **config)

def build_pairwise_model(config):
    inputs = tf.keras.Input(config["input_shape"], name="poker_input")
    
    # --- Split and combine grids ---
    inputs_A = tf.keras.Input(config["input_shape"], name="input_A")
    inputs_B = tf.keras.Input(config["input_shape"], name="input_B")
    
    hand_A = inputs_A[..., 0:1]         # (batch, 13, 4, 1)
    board_A = inputs_A[..., 1:2]        # (batch, 13, 4, 1)
    combo_A = ComboConcatLayer(name="combo_concat_A")([hand_A, board_A])

    hand_B = inputs_B[..., 0:1]         # (batch, 13, 4, 1)
    board_B = inputs_B[..., 1:2]        # (batch, 13, 4, 1)
    combo_B = ComboConcatLayer(name="combo_concat_B")([hand_B, board_B])

    encoder_cfg = get_encoder_config(config)
    encoder = PokerComboModel(encoder_cfg)

    pairwise_comparison_heads = PairwiseComparisonHeads(encoder, activation=config["activation"])
    
    hand_prob, board_prob, combined_prob = pairwise_comparison_heads([(hand_A, board_A, combo_A), (hand_B, board_B, combo_B)], training=True, return_all=True)

    # --- Build Model ---
    model = tf.keras.Model(inputs=[inputs_A, inputs_B], outputs=[hand_prob, board_prob, combined_prob])
    
    # --- Compile Model ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        loss=["mse", "mse", "mse"],
        loss_weights=config["loss_weights"],
        # loss_weights=[0.3, 0.3, 0.4],
        metrics=["mae", "mae", "mae"]
    )

    return model

# @register_keras_serializale(package="Poker")
# class PairwiseComparisonModel(tf.keras.Model):
#     """
#     Takes embeddings for A and B, and outputs sigmoid probability that A > B.
#     Uses difference vector and small MLP.
#     """
#     def __init__(self, config):
#         super().__init__(**kwargs)

#         self.hand_encoder = hand_encoder
#         self.board_encoder = board_encoder
#         self.combined_encoder = combined_encoder

#         self.rank_head = keras.Sequential([
#             layers.Dense(128, activation="relu"),
#             layers.Dense(64, activation="relu"),
#             layers.Dense(1, activation="sigmoid"),
#         ])

#     def call(self, inputs, training=False):
#         (
#             hand_A, board_A,
#             hand_B, board_B
#         ) = inputs
        
#         hA = self.hand_encoder()

#         self.embedding_dim = config["embedding_dim"]
#         self.hidden_units = config["hidden_units"]
#         self.activation = "relu"
#         self.final_activation = "sigmoid"

#         self.subtract = layers.Subtract()




# @register_keras_serializable(package="Poker")
# class PairwiseModel(tf.keras.Model):
#     """
#     Compares two combos across hand, board, and combined encoders.
#     Each encoder-head path outputs P(x1 > x2) for that representation.
#     """
#     def __init__(self, value_heads, activation="sigmoid", **kwargs):
#         """
#         Args:
#             value_heads: PokerValueHeads instance with hand_value, board_value, combined_value
#             activation: activation for final output (typically sigmoid)
#         """
#         super().__init__(**kwargs)
#         self.value_heads = value_heads

#         self.hand_diff = tf.keras.layers.Subtract()
#         self.board_diff = tf.keras.layers.Subtract()
#         self.combined_diff = tf.keras.layers.Subtract()

#         self.hand_output = tf.keras.layers.Dense(1, activation=activation, name="hand_comparison")
#         self.board_output = tf.keras.layers.Dense(1, activation=activation, name="board_comparison")
#         self.combined_output = tf.keras.layers.Dense(1, activation=activation, name="combined_comparison")

#         self.activation = activation

#         # self._pairwise_config = {"base_model_class": base_model.__class__.__name__, "activation": activation}

#     def call(self, inputs, training=False, return_all=True):
#     # def call(self, inputs, training=False, return_all=False):
#         """
#         Args:
#             inputs: tuple of (x1, x2) where each is (hand, board, combo)
#             training: whether in training mode
#             return_all: if True, return (hand_prob, board_prob, combined_prob)
#                        if False, return only combined_prob
        
#         Returns:
#             Probability that x1 > x2 for each encoder type
#         """
#         x1, x2 = inputs

#         hand1, board1, combo1 = x1        
#         hand2, board2, combo2 = x2        

#         hand_v1, board_v1, combined_v1 = self.value_heads(
#             [hand1, board1, combo1], training=training, return_all=True
#         )
#         hand_v2, board_v2, combined_v2 = self.value_heads(
#             [hand2, board2, combo2], training=training, return_all=True
#         )

#         hand_diff = self.hand_diff([hand_v1, hand_v2])
#         board_diff = self.board_diff([board_v1, board_v2])
#         combined_diff = self.combined_diff([combined_v1, combined_v2])

#         hand_prob = self.hand_output(hand_diff)
#         board_prob = self.board_output(board_diff)
#         combined_prob = self.combined_output(combined_diff)

#         # valueA = combined_v1
#         # valueB = combined_v2

#         if return_all:
#             return hand_prob, board_prob, combined_prob#, valueA, valueB
        
#         return combined_prob

#     def get_config(self):
#         base = super().get_config()
#         # base.update({"config": dict(self._pairwise_config)})        
#         base.update({
#             "value_heads_config": self.value_heads.get_config(),
#             "activation": self.activation,
#         })
#         return base

#     @classmethod
#     def from_config(cls, config):
#         activation = config.pop("activation", "sigmoid")
#         vh_cfg = config.pop("value_heads_config", None)
#         if vh_cfg == None:
#                 raise ValueError("value_heads_config is required to reconstruct PairwiseModel")
    
#         value_heads = PokerValueHeads.from_config(vh_cfg)

#         return cls(value_heads=value_heads, activation=activation, **config)

#     @property
#     def encoder_input_shape(self):
#         # Match the encoder’s expected shape from config
#         return (13, 4, 1)

# def build_pairwise_model(config: dict, encoders: Optional[PokerComboModel] = None) -> tf.keras.Model:
#     """Build a pairwise comparison model with three encoder-head paths."""

#     # Create the shared encoders and value heads
#     encoder_cfg = get_encoder_config(config)
#     if encoders is None:
#         encoders = create_encoders(encoder_cfg)
#         # encoders = PokerComboModel(encoder_cfg)
    
#     value_heads = PokerValueHeads(encoders, activation=config.get("activation", "sigmoid"))
#     pairwise_model = PairwiseModel(value_heads, activation="sigmoid")

#     dummy_input = (
#         (
#         tf.zeros((1, *config.get("input_shape_encoder", (13, 4, 1)))),
#         tf.zeros((1, *config.get("input_shape_encoder", (13, 4, 1)))),
#         tf.zeros((1, *config.get("input_shape_encoder", (13, 4, 1)))),
#         ),
#         (
#         tf.zeros((1, *config.get("input_shape_encoder", (13, 4, 1)))),
#         tf.zeros((1, *config.get("input_shape_encoder", (13, 4, 1)))),
#         tf.zeros((1, *config.get("input_shape_encoder", (13, 4, 1)))),
#         ),
#     )

#     _ = pairwise_model(dummy_input, training=False, return_all=True)

#     return pairwise_model

def create_encoders(config: dict) -> PokerComboModel:
    """Create a PokerComboModel (shared encoders) and build its sub-encoders.


    Returns a single PokerComboModel instance that contains .hand_encoder,
    .board_encoder and .combined_encoder. The returned object will be callable
    and its internal sub-encoders will be built.
    """
    enc = PokerComboModel(config)

    dummy = tf.zeros((1, *config.get("input_shape_encoder", (13, 4, 1))))
    try:
        _ = enc.hand_encoder(dummy, training=False)
        _ = enc.board_encoder(dummy, training=False)
        _ = enc.combined_encoder(dummy, training=False)
    except Exception:
        dummy_combo = tf.zeros((1, *config.get("input_shape", (13, 4, 2))))
        _ = enc(dummy_combo, training=False)

    return enc

def get_permutation_matrices(n=4):
    perms = list(itertools.permutations(range(n)))
    matrices = []
    for p in perms:
        mat = tf.one_hot(p, depth=n)
        matrices.append(mat)
    return tf.constant(tf.stack(matrices), dtype=tf.float32)

def get_encoder_config(global_config):
    return {
        "input_shape_encoder": global_config["input_shape_encoder"],
        "filters": global_config["filters"],
        "kernel_size": global_config["kernel_size"],
        "embedding_dim": global_config["embedding_dim"],
        "use_equivariance": global_config["use_equivariance"],
    }

def main():
    mat = get_permutation_matrices(4)
    print(mat)

if __name__ == "__main__":
    main()