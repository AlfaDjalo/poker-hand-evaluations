import itertools
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable
from typing import Optional


@register_keras_serializable(package="Poker")
class PokerCNNEncoder(Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        self.input_shape_encoder = config["input_shape_encoder"]  # ideally (14,4,C)
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
                "input_shape_encoder": self.input_shape_encoder,
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



@register_keras_serializable(package="Poker")
class PokerCNNEncoder_old(Model):
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
        print(f"🔍 Creating encoder with input_shape: {self.input_shape_encoder}")
        self.filters = config["filters"]
        self.kernel_size = config["kernel_size"]
        self.embedding_dim = config["embedding_dim"]
        self.use_equivariance = False #config["use_equivariance"]
        
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
        self.l2_norm = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))
        self.equiv = SuitPermutationLayer()
        # self.equiv = SuitEquivariantLayer(pooling="max")

        # Build model by calling once (if using subclassing)
        # self.build((None, *input_shape))

    def call_proposed(self, inputs, training=False):
        # inputs shape: (batch, 13, 4, channels)
        if self.use_equivariance:
            # Apply permutation layer (output: batch, 24, 13, 4, channels)
            x = self.equiv(inputs) # equiv is SuitPermutationLayer here
            # Merge batch and permutation dims to process all permuted copies at once:
            batch_size = tf.shape(x)[0]
            num_perms = tf.shape(x)[1]
            x = tf.reshape(x, (batch_size * num_perms, 13, 4, inputs.shape[-1]))
        else:
            x = inputs

        for conv, bn in zip(self.convs, self.bn):
            x = conv(x)
            x = bn(x, training=training)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.embedding(x) # shape: (batch*num_perms, embedding_dim)

        if self.use_equivariance:
            # reshape back: (batch, num_perms, embedding_dim)
            x = tf.reshape(x, (batch_size, num_perms, self.embedding_dim))

        return x


    def call(self, inputs, training=False):
        # Only apply permutation layer if use_equivariance is True
        if self.use_equivariance and self.equiv is not None:
            x = self.equiv(inputs)
        else:
            x = inputs

        for conv, bn in zip(self.convs, self.bn):
            x = conv(x)
            x = bn(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.embedding(x)
        # print("🔑 pre-norm max:", tf.reduce_max(tf.abs(x)))
        x = self.l2_norm(x)
        # print("🔑 post-norm norm:", tf.norm(x))
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
    def encoder_input_shape(self, config):
        # Match the encoder’s expected shape from config
        return config["input_shape_encoder"] #(13, 4, 2)

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
class PokerCategoryModel(Model):
    """
    Model to convert embedding into hand category
    
    Inputs:
        encoder model

    Outputs:
        hand category: ***integer***
    """
    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.category_head = layers.Dense(9, activation="softmax")
        self._category_config = {"encoder_class": encoder.__class__.__name__}

    def call(self, inputs, training=False):
        embedding = self.encoder(inputs, training=training)
        category = self.category_head(embedding)
        return category

    def get_config(self):
        base = super().get_config()
        base.update({"config": dict(self._category_config)})
        return base

    @classmethod
    def from_config(cls, config):
        cfg = config.pop("config", {})
        return cls(encoder=None, activation=cfg.get("activation", "softmax"), **config)

    @property
    def encoder_input_shape(self):
        return self.encoder.input_shape_encoder

@register_keras_serializable(package="Poker")
class PokerCategoryHeads(tf.keras.Model):
    """
    Model to create encoder/value models for hand, board and combined.
    """
    def __init__(self, encoders=None, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        if encoders == None:
            raise ValueError("PokerValueHeads requires encoders parameter at construction time.")
        self.encoders = encoders
        self.hand_category = PokerCategoryModel(self.encoders.hand_encoder)
        self.board_category = PokerCategoryModel(self.encoders.board_encoder)
        self.combined_category = PokerCategoryModel(self.encoders.combined_encoder)
        self.activation = activation

    def call(self, inputs, training=False, return_all=True):
    # def call(self, inputs, training=False, return_all=False):
        hand, board, combo = inputs
        print(f"🔍 Combo shape received: {combo.shape}")

        hand_v = self.hand_category(hand, training=training)
        board_v = self.board_category(board, training=training)
        combined_v = self.combined_category(combo, training=training)

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
class SuitPermutationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.perms = list(itertools.permutations(range(4)))

    def call(self, inputs):
        # inputs: (batch, 13, 4, channels)
        permuted = []
        for perm in self.perms:
            permuted.append(tf.gather(inputs, indices=list(perm), axis=2))
        # Stack on new axis for permutations
        return tf.stack(permuted, axis=1) # shape: (batch, 24, 13, 4, channels)
    
    def get_config(self):
        return super().get_config()


@register_keras_serializable(package="Poker")
class SuitEquivariantLayer(tf.keras.layers.Layer):
    """
    Make representation invariant to relabelling of suits (global permutation).
    For each of the 24 suit permutations we permute the suit axis, compute
    the permuted inputs, then pool (max) across permutations. This yields
    a representation invariant to global relabelling of suits while
    preserving suit-structure (suited vs offsuit).
    """
    def __init__(self, pooling="max", **kwargs):
        super().__init__(**kwargs)
        if pooling not in ("max", "mean"):
            raise ValueError("pooling must be 'max' or 'mean'")
        self.pooling = pooling
        # store permutations as a Python list of index tuples for clarity
        self.perms = list(itertools.permutations(range(4)))

    def call(self, inputs):
        # Expect inputs shape: (batch, 13, 4, channels)
        # We'll permute axis=2 (suit axis)
        # Build a list of permuted tensors, one per permutation
        # Each permuted tensor has shape (batch, 13, 4, channels)
        permuted = []
        for perm in self.perms:
            permuted.append(tf.gather(inputs, indices=list(perm), axis=2))

        # Stack: shape (batch, 24, 13, 4, channels)
        X_perm = tf.stack(permuted, axis=1)

        # Pool across the permutation axis to enforce invariance.
        if self.pooling == "max":
            X_pooled = tf.reduce_max(X_perm, axis=1)
        else:
            X_pooled = tf.reduce_mean(X_perm, axis=1)

        # Result shape: (batch, 13, 4, channels)
        return X_pooled

    def get_config(self):
        base = super().get_config()
        base.update({"pooling": self.pooling})
        return base

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable(package="Poker")
class SuitEquivariantLayer_old(tf.keras.layers.Layer):
    """
    Make representation invariant to relabelling of suits (global permutation).
    For each of the 24 suit permutations we permute the suit axis, compute
    the permuted inputs, then pool (max) across permutations. This yields
    a representation invariant to global relabelling of suits while
    preserving suit-structure (suited vs offsuit).
    """
    def __init__(self, pooling="max", **kwargs):
        super().__init__(**kwargs)
        if pooling not in ("max", "mean"):
            raise ValueError("pooling must be 'max' or 'mean'")
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
        super().__init__(**kwargs)
        self._encoder_config = dict(get_encoder_config(config)) if isinstance(config, dict) else dict(config)
        self.use_shared_encoder = config.get("use_shared_encoder", False)  # NEW

        if self.use_shared_encoder:
            self.shared_encoder = PokerCNNEncoder(self._encoder_config)  

            self.hand_encoder = self.shared_encoder
            self.board_encoder = self.shared_encoder
            self.combined_encoder = self.shared_encoder
        else:
            self.hand_encoder = PokerCNNEncoder(self._encoder_config)
            self.board_encoder = PokerCNNEncoder(self._encoder_config)
            self.combined_encoder = PokerCNNEncoder(self._encoder_config)

        print(f"🔍 PokerComboModel encoder config: {self._encoder_config}")
        print(f"🔍 Using {'SHARED' if self.use_shared_encoder else 'SEPARATE'} encoder(s)")

    def call(self, inputs, training=False, return_all=True):
        # inputs: (batch, 13, 4, 2)
        hand = inputs[..., 0:1] # (batch, 13, 4, 1)
        board = inputs[..., 1:2] # (batch, 13, 4, 1)
        combined  = hand + board # (batch, 13, 4, 1)

        # Feed to encoder
        hand_emb = self.hand_encoder(hand, training=training)
        board_emb = self.board_encoder(board, training=training)
        combined_emb = self.combined_encoder(combined, training=training)

        if return_all:
            return hand_emb, board_emb, combined_emb

        return combined_emb
    
    def get_config(self):
        base = super().get_config()
        config_copy = dict(self._encoder_config)
        config_copy["use_shared_encoder"] = self.use_shared_encoder
        base.update({"config": config_copy})
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

def build_value_model(config):
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
    combo = ComboConcatLayer(name="combo_concat")([hand, board])

    encoder_config = get_encoder_config(config)
    encoder = PokerComboModel(encoder_config)
    value_heads = PokerValueHeads(encoder, activation=config["activation"])
    hand_v, board_v, combined_v = value_heads([hand, board, combo], training=True, return_all=True)

    # --- Build Model ---
    model = tf.keras.Model(inputs=inputs, outputs=[hand_v, board_v, combined_v])
    
    # --- Compile Model ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        loss=["mse", "mse", "mse"],
        loss_weights=config["loss_weights"],
        metrics=["mae", "mae", "mae"]
    )

    return model

def build_category_model(config):
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
    combo = ComboConcatLayer(name="combo_concat")([hand, board])

    encoder_config = get_encoder_config(config)
    encoder = PokerComboModel(encoder_config)
    category_heads = PokerCategoryHeads(encoder)
    hand_v, board_v, combined_v = category_heads([hand, board, combo], training=True, return_all=True)

    # --- Build Model ---
    model = tf.keras.Model(inputs=inputs, outputs=[hand_v, board_v, combined_v])
    
    # --- Compile Model ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        loss=[tf.keras.losses.CategoricalCrossentropy()] * 3,      # <- use categorical for one-hot targets
        loss_weights=config["loss_weights"],
        metrics=[tf.keras.metrics.CategoricalAccuracy()] * 3      # <- categorical accuracy
    )

    return model

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

    encoder_config = get_encoder_config(config)
    encoder = PokerComboModel(encoder_config)

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


def create_encoders(config: dict) -> PokerComboModel:
    """Create a PokerComboModel (shared encoders) and build its sub-encoders.


    Returns a single PokerComboModel instance that contains .hand_encoder,
    .board_encoder and .combined_encoder. The returned object will be callable
    and its internal sub-encoders will be built.
    """
    enc = PokerComboModel(config)

    dummy = tf.zeros((1, *config.get("input_shape_encoder", (14, 4, 1))))
    # dummy = tf.zeros((1, *config.get("input_shape_encoder", (13, 4, 1))))
    try:
        _ = enc.hand_encoder(dummy, training=False)
        _ = enc.board_encoder(dummy, training=False)
        _ = enc.combined_encoder(dummy, training=False)
    except Exception:
        dummy_combo = tf.zeros((1, *config.get("input_shape", (14, 4, 2))))
        # dummy_combo = tf.zeros((1, *config.get("input_shape", (13, 4, 2))))
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
        "use_shared_encoder": global_config.get("use_shared_encoder", False),
    }


def one_hot_grid_for_cards(cards, mode="combined"):
    # helper similar to your cards_to_tensor but produces (13,4,1) grid for a single set
    rank_to_idx = {'A':0,'K':1,'Q':2,'J':3,'T':4,'9':5,'8':6,'7':7,'6':8,'5':9,'4':10,'3':11,'2':12}
    suit_to_idx = {'s':0,'h':1,'d':2,'c':3}
    grid = np.zeros((14,4,1), dtype=np.float32)
    # grid = np.zeros((13,4,1), dtype=np.float32)
    for card in cards:
        r = card[0].upper(); s = card[-1].lower()
        if r in rank_to_idx and s in suit_to_idx:
            grid[rank_to_idx[r], suit_to_idx[s], 0] = 1.0
    return tf.constant(grid)


def main():
    # mat = get_permutation_matrices(4)
    # print(mat)

    # hands:
    JhTh = one_hot_grid_for_cards(["Jh","Th"])   # suited hearts
    JdTd = one_hot_grid_for_cards(["Jd","Td"])   # suited diamonds (should match JhTh)
    JhTd = one_hot_grid_for_cards(["Jh","Td"])   # offsuit (should differ)

    batch = tf.stack([JhTh, JdTd, JhTd], axis=0)  # shape (3,13,4,1)

    layer = SuitEquivariantLayer(pooling="max")
    out = layer(batch)  # shape (3,13,4,1)

    # flatten embeddings per sample to compare (or pass through a tiny conv + flatten to mimic encoder)
    flat = tf.reshape(out, (3, -1)).numpy()

    # pairwise distances
    d01 = np.linalg.norm(flat[0] - flat[1])
    d02 = np.linalg.norm(flat[0] - flat[2])

    print("dist JhTh-JdTd:", d01)
    print("dist JhTh-JhTd:", d02)


if __name__ == "__main__":
    main()