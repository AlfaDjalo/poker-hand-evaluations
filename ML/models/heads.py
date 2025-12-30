import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Poker")
class CombinedInputValueHead(Model):
    """
    Value head that operates directly on combined embedding.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value_head = layers.Dense(1, activation="sigmoid")

    def __call__(self, inputs, training=False):
        combo_emb = inputs["combo_emb"]
        return self.value_head(combo_emb)

    def get_config(self):
        return super().get_config()
    

@register_keras_serializable(package="Poker")
class SeparateInputValueHead(Model):
    """
    Value head that operates directly on hand and board embedding.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value_head = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        hand_emb = inputs["hand_emb"]
        board_emb = inputs["board_emb"]
        stacked_emb = tf.keras.layers.Concatenate(name="hand_board_concat")([hand_emb, board_emb])
        # stacked_emb = tf.keras.layers.Concatenate(name="hand_board_concat")(inputs)
        return self.value_head(stacked_emb)

    def get_config(self):
        return super().get_config()


@register_keras_serializable(package="Poker")
class CombinedInputPairwiseComparisonHead(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.concat = layers.Concatenate(axis=-1)
        self.pairwise_head = layers.Dense(1, activation="sigmoid")

    def __call__(self, inputs, training=False):
        embedding_A = inputs["combo_emb_A"]
        embedding_B = inputs["combo_emb_B"]
        combined_embeddings = self.concat([embedding_A, embedding_B])
        return self.pairwise_head(combined_embeddings)


@register_keras_serializable(package="Poker")
class SeparateInputPairwiseComparisonHead(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.concat = layers.Concatenate(axis=-1)
        self.pairwise_head = layers.Dense(1, activation="sigmoid")

    def __call__(self, inputs, training=False):
        hand_embedding_A = inputs["hand_emb_A"]
        hand_embedding_B = inputs["hand_emb_B"]
        board_embedding_A = inputs["board_emb_A"]
        board_embedding_B = inputs["board_emb_B"]
        combined_embedding_A = self.concat([hand_embedding_A, board_embedding_A])
        combined_embedding_B = self.concat([hand_embedding_B, board_embedding_B])
        stacked_emb = self.concat([combined_embedding_A, combined_embedding_B])
        return self.pairwise_head(stacked_emb)


@register_keras_serializable(package="Poker")
class CombinedInputHandCategoryHead(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category_head = layers.Dense(9, activation="softmax")

    def __call__(self, inputs, training=False):
        combo_emb = inputs["combo_emb"]
        return self.category_head(combo_emb)


@register_keras_serializable(package="Poker")
class SeparateInputHandCategoryHead(Model):
    """
    Category head that operates directly on hand and board embedding.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category_head = layers.Dense(9, activation="softmax")

    def __call__(self, inputs, training=False):
        hand_emb = inputs["hand_emb"]
        board_emb = inputs["board_emb"]
        stacked_emb = tf.keras.layers.Concatenate(name="hand_board_concat")([hand_emb, board_emb])
        # stacked_emb = tf.keras.layers.Concatenate(name="hand_board_concat")(inputs)
        return self.category_head(stacked_emb)

    def get_config(self):
        return super().get_config()


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

