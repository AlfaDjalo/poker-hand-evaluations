# reinforce_trainer.py
import tensorflow as tf

def policy_loss(p_push, actions, rewards):
    actions = tf.cast(actions, tf.float32)
    rewards = tf.cast(rewards, tf.float32)

    logp = (
        actions * tf.math.log(p_push + 1e-8)
        + (1 - actions) * tf.math.log(1 - p_push + 1e-8)
    )
    return -tf.reduce_mean(logp * rewards)

@tf.function
def train_step(policy, optimizer, batch):
    with tf.GradientTape() as tape:
        p_push = policy(batch["obs"], training=True)
        loss = policy_loss(p_push, batch["actions"], batch["rewards"])

    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))
    return loss
