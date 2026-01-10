import tensorflow as tf

class PushFoldPolicy(tf.keras.Model):
    def __init__(
        self,
        hidden_dim_1=64,
        hidden_dim_2=32,
        num_actions=2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.num_actions = num_actions

        self.shared_1 = tf.keras.layers.Dense(
            hidden_dim_1,
            activation="relu"
        )
        self.shared_2 = tf.keras.layers.Dense(
            hidden_dim_2,
            activation="relu"
        )

        self.policy_logits = tf.keras.layers.Dense(num_actions)
        self.value = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.shared_1(inputs)
        x = self.shared_2(x)
        return {
            "logits": self.policy_logits(x),
            "value": tf.squeeze(self.value(x), axis=-1)
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim_1": self.hidden_dim_1,
            "hidden_dim_2": self.hidden_dim_2,
            "num_actions": self.num_actions
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PushFoldPolicyOld(tf.keras.Model):
    def __init__(self, hidden_dim=32, num_actions=2, **kwargs):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.shared = tf.keras.layers.Dense(
            hidden_dim,
            activation="relu"
        )

        self.policy_logits = tf.keras.layers.Dense(num_actions)
        self.value = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.shared(inputs)
        return {
            "logits": self.policy_logits(x),
            "value": tf.squeeze(self.value(x), axis=-1)
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_actions": self.num_actions
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
