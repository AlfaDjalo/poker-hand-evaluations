# pushfold_agent.py
import tensorflow as tf
import numpy as np

class PushFoldAgent:
    def __init__(self, policy, training=True):
        self.policy = policy
        self.training = training

    # def act(self, obs):
    #     embedding = tf.expand_dims(tf.convert_to_tensor(obs["embedding"], dtype=tf.float32), axis=0)
    #     logits = self.policy(embedding)
    #     action_probs = tf.nn.softmax(logits).numpy()[0]
    #     # print("Probs: ", action_probs)
    #     return np.random.choice(len(action_probs), p=action_probs)
    
    def act(self, obs):
        embedding = tf.expand_dims(
            tf.convert_to_tensor(obs["embedding"], dtype=tf.float32),
            axis=0
        )

        out = self.policy(embedding, training=False)
        logits = out["logits"]

        action_probs = tf.nn.softmax(logits).numpy()[0]
        return np.random.choice(len(action_probs), p=action_probs)
    
    # def action_probs(self, obs):
    #     embedding = tf.expand_dims(tf.convert_to_tensor(obs["embedding"], dtype=tf.float32), axis=0)
    #     logits = self.policy(embedding)
    #     action_probs = tf.nn.softmax(logits).numpy()[0]
    #     return action_probs

    def policy_probs(self, embedding):
        out = self.policy(
            tf.expand_dims(embedding, axis=0),
            training=False
        )
        logits = out["logits"]
        return tf.nn.softmax(logits, axis=-1).numpy()[0]
    
    
    def policy_values(self, embedding):
        out = self.policy(
            tf.expand_dims(embedding, axis=0),
            training=False
        )
        values = out["value"]
        return values