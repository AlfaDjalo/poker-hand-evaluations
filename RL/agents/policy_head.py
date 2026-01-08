import tensorflow as tf

class PushFoldPolicy(tf.keras.Model):
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
    

class PushFoldPolicyOld(tf.keras.Model):
    def __init__(self, num_actions=2, **kwargs):
        super().__init__(**kwargs)
        self.num_actions = num_actions
        
        self.policy_logits = tf.keras.layers.Dense(num_actions, activation=None)

        # Value head (scalar)
        self.value = tf.keras.layers.Dense(1, activation=None)

    # def call(self, inputs, training=False):
    #     return self.policy_logits(inputs)

    def call(self, inputs, training=False):
        x = self.shared(inputs)
        return {
            "logits": self.policy_logits(x),
            "value": tf.squeeze(self.value(x), axis=-1)
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_actions": self.num_actions
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    