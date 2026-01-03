import tensorflow as tf
import numpy as np
from RL.env.push_fold_env import PushFoldEnv
from RL.agents.policy_head import PushFoldPolicy

SB_MODEL_PATH = "models/saved/policies/push_fold_policy_sb.keras"
BB_MODEL_PATH = "models/saved/policies/push_fold_policy_bb.keras"

def select_action(policy, obs, deterministic=False):
    emb = obs["embedding"]
    logits = policy(tf.convert_to_tensor([emb], dtype=tf.float32))
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    print("Probs: ", probs)
    if deterministic:
        return int(np.argmax(probs))
    else:
        return int(np.random.choice(len(probs), p=probs))

def run_test(num_episodes=1000, deterministic=False):
    env = PushFoldEnv({
        "sb": 0.5,
        "bb": 1.0,
        "stack_bb": 10,
        "allow_variable_stack": False
    })

    sb_policy = tf.keras.models.load_model(
        SB_MODEL_PATH,
        custom_objects={"PushFoldPolicy": PushFoldPolicy},
        compile=False
    )

    bb_policy = tf.keras.models.load_model(
        BB_MODEL_PATH,
        custom_objects={"PushFoldPolicy": PushFoldPolicy},
        compile=False
    )

    sb_reward_total = 0.0

    for _ in range(num_episodes):
        obs = env.reset()
        done = False

        sb_action = select_action(sb_policy, obs, deterministic)
        obs, reward, done, info = env.step(sb_action)

        if not done:
            bb_action = select_action(bb_policy, obs, deterministic)
            obs, reward, done, info = env.step(bb_action)

        sb_reward_total += reward


        # while not done:
        #     action = select_action(policy, obs, deterministic)
        #     obs, reward, done, info = env.step(action)

            # if done:
            #     if info["winner"] == env.POS_SB:
            #         sb_reward_total += env.pot - env.committed[env.POS_SB]
            #     else:
            #         sb_reward_total -= env.committed[env.POS_SB]

    ev = sb_reward_total / num_episodes
    print(f"SB EV over {num_episodes} episodes: {ev:.3f} BB")

if __name__ == "__main__":
    run_test(num_episodes=100, deterministic=False)
    run_test(num_episodes=100, deterministic=True)
