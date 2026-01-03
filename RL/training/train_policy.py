import tensorflow as tf
import numpy as np
from RL.env.push_fold_env import PushFoldEnv
from RL.agents.policy_head import PushFoldPolicy
from RL.agents.pushfold_agent import PushFoldAgent

model_path = "models/saved/policies/push_fold_policy.keras"

# Simple policy gradient loss
def compute_loss(logits, actions, rewards, entropy_coef=0.02):
    # Policy loss
    neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions,
        logits=logits
    )
    policy_loss = tf.reduce_mean(neg_log_prob * rewards)

    # Entropy bonus
    probs = tf.nn.softmax(logits)
    entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1))

    # Total loss
    loss = policy_loss - entropy_coef * entropy
    return loss

# def compute_loss(logits, actions, rewards):
#     neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
#     loss = tf.reduce_mean(neg_log_prob * rewards)
#     return loss

# @tf.function
def train_step(policy, optimizer, observations, actions, rewards):
    with tf.GradientTape() as tape:
        logits = policy(observations, training=True)
        loss = compute_loss(logits, actions, rewards)
    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))
    return loss


def main(num_episodes=200):
    env = PushFoldEnv({
        "sb": 0.5,
        "bb": 1.0,
        "stack_bb": 10,
        "allow_variable_stack": False
    })

    sb_policy = PushFoldPolicy(embedding_dim=32)
    bb_policy = PushFoldPolicy(embedding_dim=32)

    # dummy_input = tf.zeros((1, 32), dtype=tf.float32)  # assuming embedding_dim=32
    # sb_policy(dummy_input)
    # bb_policy(dummy_input)

    sb_agent = PushFoldAgent(sb_policy)
    bb_agent = PushFoldAgent(bb_policy)

    sb_optimizer = tf.keras.optimizers.Adam(1e-3)
    bb_optimizer = tf.keras.optimizers.Adam(1e-3)

    for episode in range(num_episodes):
        obs = env.reset()
        done = False

        # ---------- SB acts ----------
        sb_obs = obs["embedding"]
        sb_action = sb_agent.act(obs)

        obs, reward, done, info = env.step(sb_action)

        # ---------- BB may act ----------
        bb_obs = None
        bb_action = None

        if not done:
            bb_obs = obs["embedding"]
            bb_action = bb_agent.act(obs)
            obs, reward, done, info = env.step(bb_action)

        # ---------- terminal reward ----------
        # reward is SB payoff
        sb_reward = reward
        bb_reward = -reward if bb_action is not None else None

        # ---------- train SB ----------
        sb_loss = train_step(
            sb_policy,
            sb_optimizer,
            tf.convert_to_tensor([sb_obs], dtype=tf.float32),
            tf.convert_to_tensor([sb_action], dtype=tf.int32),
            tf.convert_to_tensor([sb_reward], dtype=tf.float32),
        )

        # ---------- train BB (only if BB acted) ----------
        if bb_action is not None:
            bb_loss = train_step(
                bb_policy,
                bb_optimizer,
                tf.convert_to_tensor([bb_obs], dtype=tf.float32),
                tf.convert_to_tensor([bb_action], dtype=tf.int32),
                tf.convert_to_tensor([bb_reward], dtype=tf.float32),
            )
        else:
            bb_loss = None

        if episode % 10 == 0:
            print(
                f"Ep {episode:4d} | "
                f"SB loss {sb_loss.numpy(): .4f} | "
                f"BB loss {bb_loss.numpy(): .4f}" if bb_loss is not None else
                f"Ep {episode:4d} | SB loss {sb_loss.numpy(): .4f}"
            )

    sb_policy.save("models/saved/policies/push_fold_policy_sb.keras")
    bb_policy.save("models/saved/policies/push_fold_policy_bb.keras")


def main_old(num_episodes=10, load_model=False):
    # Setup environment and agent
    env = PushFoldEnv({
        "sb": 0.5,
        "bb": 1.0,
        "stack_bb": 10,
        "allow_variable_stack": False
    })

    # if load_model == True:
    #     policy = tf.keras.models.load_model(
    #         model_path,
    #         custom_objects={"PushFoldPolicy": PushFoldPolicy},
    #         compile=False
    #     )
    # else:
    #     policy = PushFoldPolicy(embedding_dim=32)

    sb_policy = PushFoldPolicy(embedding_dim=32)
    bb_policy = PushFoldPolicy(embedding_dim=32)

    sb_agent = PushFoldAgent(sb_policy)
    bb_agent = PushFoldAgent(bb_policy)

    sb_optimizer = tf.keras.optimizers.Adam(1e-3)
    bb_optimizer = tf.keras.optimizers.Adam(1e-3)


    # encoder_dim = 32  # get encoder output size
    # policy = PushFoldPolicy(embedding_dim=encoder_dim)
    # agent = PushFoldAgent(policy)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # num_episodes = 100  # just a small number for testing

    for episode in range(num_episodes):
        obs = env.reset()
        done = False

        sb_observations = []
        sb_actions = []
        sb_rewards = []

        bb_observations = []
        bb_actions = []
        bb_rewards = []

        # SB action
        sb_action = sb_agent.act(obs)
        next_obs, reward, done, info = env.step(sb_action)

        # Save data for training
        sb_observations.append(obs["embedding"])
        sb_actions.append(sb_action)
        sb_rewards.append(reward if reward != 0 else 0)

        obs = next_obs

        # BB action, if necessary
        if not done:
            bb_action = bb_agent.act(obs)
            next_obs, reward, done, info = env.step(bb_action)

            # Save data for training
            bb_observations.append(obs["embedding"])
            bb_actions.append(bb_action)
            bb_rewards.append(reward if reward != 0 else 0)

            # Showdown, if necessary
            if bb_action == 1:
                sb_rewards.append(-reward)
        
        # while not done:
        #     action = agent.act(obs)
        #     next_obs, reward, done, info = env.step(action)

        #     # Save data for training
        #     observations.append(obs["embedding"])
        #     actions.append(action)
        #     rewards.append(reward if reward != 0 else 0)

        #     obs = next_obs

        # Convert lists to tensors for training
        sb_observations = tf.convert_to_tensor(sb_observations, dtype=tf.float32)
        sb_actions = tf.convert_to_tensor(sb_actions, dtype=tf.int32)
        sb_rewards = tf.convert_to_tensor(sb_rewards, dtype=tf.float32)

        bb_observations = tf.convert_to_tensor(bb_observations, dtype=tf.float32)
        bb_actions = tf.convert_to_tensor(bb_actions, dtype=tf.int32)
        bb_rewards = tf.convert_to_tensor(bb_rewards, dtype=tf.float32)

        sb_loss = train_step(sb_policy, sb_optimizer, sb_observations, sb_actions, sb_rewards)
        bb_loss = train_step(bb_policy, bb_optimizer, bb_observations, bb_actions, bb_rewards)

        if episode % 10 == 0:
            print(f"Episode {episode+1} loss: {loss.numpy():.4f}")
        # print(f"Episode {episode+1} loss: {loss.numpy():.4f}")

    # After training loop ends
    sb_policy.save("models/saved/policies/push_fold_policy_sb.keras")
    bb_policy.save("models/saved/policies/push_fold_policy_bb.keras")
    print("Models saved to models/saved/policies/push_fold_policy_[pos].keras")


if __name__ == "__main__":
    main(1000)