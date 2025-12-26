import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Poker")
class PokerValueModel(Model):
    def __init__(self, encoder, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.value_head = layers.Dense(1, activation=activation)
        self._value_config = {"activation": activation, "encoder_class": encoder.__class__.__name__}

    def call(self, inputs, training=False):
        embedding = self.encoder(inputs, training=training)
        return self.value_head(embedding)

    def get_config(self):
        base = super().get_config()
        base.update({"config": dict(self._value_config)})
        return base

@register_keras_serializable(package="Poker")
class PokerValueHeads(tf.keras.Model):
    # ...same as before but imported here...
    def __init__(self, encoders=None, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        if encoders is None:
            raise ValueError("PokerValueHeads requires encoders parameter at construction time.")
        self.encoders = encoders
        self.hand_value = PokerValueModel(self.encoders.hand_encoder, activation)
        self.board_value = PokerValueModel(self.encoders.board_encoder, activation)
        self.combined_value = PokerValueModel(self.encoders.combined_encoder, activation)
        self.activation = activation

    def call(self, inputs, training=False, return_all=True):
        hand, board, combo = inputs
        hand_v = self.hand_value(hand, training=training)
        board_v = self.board_value(board, training=training)
        combined_v = self.combined_value(combo, training=training)
        if return_all:
            return hand_v, board_v, combined_v
        return combined_v

@register_keras_serializable(package="Poker")
class PokerCategoryModel(Model):
    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.category_head = layers.Dense(9, activation="softmax")
        self._category_config = {"encoder_class": encoder.__class__.__name__}

    def call(self, inputs, training=False):
        embedding = self.encoder(inputs, training=training)
        return self.category_head(embedding)

@register_keras_serializable(package="Poker")
class PokerCategoryHeads(tf.keras.Model):
    def __init__(self, encoders=None, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        if encoders is None:
            raise ValueError("PokerCategoryHeads requires encoders parameter at construction time.")
        self.encoders = encoders
        self.hand_category = PokerCategoryModel(self.encoders.hand_encoder)
        self.board_category = PokerCategoryModel(self.encoders.board_encoder)
        self.combined_category = PokerCategoryModel(self.encoders.combined_encoder)
        self.activation = activation

    def call(self, inputs, training=False, return_all=True):
        hand, board, combo = inputs
        hand_v = self.hand_category(hand, training=training)
        board_v = self.board_category(board, training=training)
        combined_v = self.combined_category(combo, training=training)
        if return_all:
            return hand_v, board_v, combined_v
        return combined_v

@register_keras_serializable(package="Poker")
class PairwiseComparisonModel(Model):
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
        return self.pairwise_head(combined_embeddings)

@register_keras_serializable(package="Poker")
class PairwiseComparisonHeads(tf.keras.Model):
    def __init__(self, encoders=None, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        if encoders is None:
            raise ValueError("PairwiseComparisonHeads requires encoders parameter at construction time.")
        self.encoders = encoders
        self.hand_comparison = PairwiseComparisonModel(self.encoders.hand_encoder, activation)
        self.board_comparison = PairwiseComparisonModel(self.encoders.board_encoder, activation)
        self.combined_comparison = PairwiseComparisonModel(self.encoders.combined_encoder, activation)
        self.activation = activation

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

class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_weights, from_logits=False, name="weighted_cce"):
        super().__init__(name=name)
        self.class_weights = tf.reshape(class_weights, (1, -1))
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        weights = tf.reduce_sum(self.class_weights * y_true, axis=-1)
        return ce * weights
