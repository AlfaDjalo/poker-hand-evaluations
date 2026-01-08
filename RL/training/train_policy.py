import tensorflow as tf
import numpy as np
import time

from RL.env.push_fold_env import PushFoldEnv
from RL.agents.policy_head import PushFoldPolicy
from RL.agents.pushfold_agent import PushFoldAgent
from RL.agents.all_in_agent import AllInAgent

sb_model_path = "models/saved/policies/push_fold_policy_sb.keras"
bb_model_path = "models/saved/policies/push_fold_policy_bb.keras"

# Simple policy gradient loss
def compute_loss_old(logits, actions, rewards, entropy_coef=0.01):
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


def compute_loss(logits, values, actions, rewards,
                 value_coef=0.5, entropy_coef=0.01):

    # --- Policy loss ---
    neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions,
        logits=logits
    )

    advantages = rewards - values
    policy_loss = tf.reduce_mean(neg_log_prob * tf.stop_gradient(advantages))

    # --- Value loss ---
    value_loss = tf.reduce_mean(tf.square(advantages))

    # --- Entropy bonus ---
    probs = tf.nn.softmax(logits)
    entropy = -tf.reduce_mean(
        tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
    )

    total_loss = (
        policy_loss
        + value_coef * value_loss
        - entropy_coef * entropy
    )

    return total_loss, policy_loss, value_loss

def train_step(policy, optimizer, observations, actions, rewards):
    with tf.GradientTape() as tape:
        out = policy(observations, training=True)
        logits = out["logits"]
        values = out["value"]

        loss, policy_loss, value_loss = compute_loss(
            logits, values, actions, rewards
        )

    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))

    return loss, policy_loss, value_loss


# @tf.function
def train_step_old(policy, optimizer, observations, actions, rewards):
    with tf.GradientTape() as tape:
        logits = policy(observations, training=True)
        loss = compute_loss(logits, actions, rewards)
    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))
    return loss


def main(num_episodes=200, batch_size=64, load_sb_model=False, load_bb_model=False):


    # sb_loss = tf.constant(0.0)
    # sb_policy_loss = tf.constant(0.0)
    # sb_value_loss = tf.constant(0.0)

    start_time = time.time()

    env = PushFoldEnv({
        "sb": 0.5,
        "bb": 1.0,
        "stack_bb": 10,
        "allow_variable_stack": False
    })

    if load_sb_model == True:
        sb_policy = tf.keras.models.load_model(
            sb_model_path,
            custom_objects={"PushFoldPolicy": PushFoldPolicy},
            compile=False
        )
    else:
        sb_policy = PushFoldPolicy(num_actions=2)

    if load_bb_model == True:
        bb_policy = tf.keras.models.load_model(
            bb_model_path,
            custom_objects={"PushFoldPolicy": PushFoldPolicy},
            compile=False
        )
    else:
        bb_policy = PushFoldPolicy(num_actions=2)

    sb_agent = PushFoldAgent(sb_policy)
    # bb_agent = PushFoldAgent(bb_policy)
    bb_agent = AllInAgent()

    sb_optimizer = tf.keras.optimizers.Adam(1e-4)
    bb_optimizer = tf.keras.optimizers.Adam(1e-4)

    batch_obs = []
    batch_actions = []
    batch_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False

        # ---------- SB acts ----------
        sb_obs = obs["embedding"]
        sb_action = sb_agent.act(obs)

        sb_transition = (sb_obs, sb_action)

        obs, reward, done, info = env.step(sb_action)

        # Get SB's actual reward
        sb_reward = info.get('sb_reward', 0.0)

        batch_obs.append(sb_obs)
        batch_actions.append(sb_action)
        batch_rewards.append(sb_reward)

        # ---------- BB may act ----------
        bb_obs = None
        bb_action = None

        if not done:
            bb_obs = obs["embedding"]
            bb_action = bb_agent.act(obs)
            obs, reward, done, info = env.step(bb_action)
            sb_reward = info['sb_reward']  # Update with final reward
        else:
            info = {"winner": env.POS_BB}

        # sb_reward = reward

        # if bb_action is not None:
        #     if bb_action == 0:  # BB FOLDS
        #         bb_reward = -1.0
        #     else:               # BB CALLS
        #         bb_reward = -reward
        # else:
        #     bb_reward = None

        # print("SB reward: ", sb_reward, " BB reward: ", bb_reward)

        # Normalize
        # sb_reward = sb_reward / env.stack_bb
        # bb_reward = bb_reward / env.stack_bb
        
        # ---------- train SB ----------
        if len(batch_obs) >= batch_size:
            obs_tensor = tf.convert_to_tensor(batch_obs, dtype=tf.float32)
            actions_tensor = tf.convert_to_tensor(batch_actions, dtype=tf.int32)
            rewards_tensor = tf.convert_to_tensor(batch_rewards, dtype=tf.float32)
            
            sb_loss, sb_policy_loss, sb_value_loss = train_step(sb_policy, sb_optimizer, obs_tensor, actions_tensor, rewards_tensor)

            # Clear batches
            batch_obs.clear()
            batch_actions.clear()
            batch_rewards.clear()

        # sb_loss, sb_policy_loss, sb_value_loss = train_step(
        #     sb_policy,
        #     sb_optimizer,
        #     tf.convert_to_tensor([sb_obs], dtype=tf.float32),
        #     tf.convert_to_tensor([sb_action], dtype=tf.int32),
        #     tf.convert_to_tensor([sb_reward], dtype=tf.float32),
        # )

        # ---------- train BB (only if BB acted) ----------
        # if bb_action is not None:
        #     bb_loss = train_step(
        #         bb_policy,
        #         bb_optimizer,
        #         tf.convert_to_tensor([bb_obs], dtype=tf.float32),
        #         tf.convert_to_tensor([bb_action], dtype=tf.int32),
        #         tf.convert_to_tensor([bb_reward], dtype=tf.float32),
        #     )
        # else:
        #     bb_loss = None

        if episode % 50 == 0 and episode > batch_size:
        # if episode % 50 == 0:
            print(
                f"Ep {episode:4d} | "
                f"SB total {sb_loss.numpy(): .4f} | "
                f"policy {sb_policy_loss.numpy(): .4f} | "
                f"value {sb_value_loss.numpy(): .4f}"
            )

        # if episode % 100 == 0:
        #     print(
        #         f"Ep {episode:4d} | "
        #         f"SB loss {sb_loss.numpy(): .4f} | "
        #         # f"BB loss {bb_loss.numpy(): .4f}" if bb_loss is not None else
        #         # f"Ep {episode:4d} | SB loss {sb_loss.numpy(): .4f}"
        #     )

    sb_policy.save(sb_model_path)
    # bb_policy.save(bb_model_path)

    print("Models saved to models/saved/policies/push_fold_policy_[pos].keras")

    end_time = time.time()

    print("Time taken: ", end_time - start_time)


if __name__ == "__main__":
    main(30000, load_sb_model=True, load_bb_model=False)