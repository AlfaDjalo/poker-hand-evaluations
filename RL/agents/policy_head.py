import tensorflow as tf

class PushFoldPolicy(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim=32, num_actions=2, **kwargs):
        super().__init__(**kwargs) 
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.logits = tf.keras.layers.Dense(num_actions)  # output logits for softmax

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.logits(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_actions": self.num_actions
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)