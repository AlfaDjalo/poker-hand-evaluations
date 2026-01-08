import random
import tensorflow as tf

from RL.env.push_fold_env import PushFoldEnv

from RL.agents.random_agent import RandomAgent
from RL.agents.all_in_agent import AllInAgent
from RL.agents.SB_always_fold_agent import SBAlwaysFoldAgent
from RL.agents.BB_always_fold_agent import BBAlwaysFoldAgent
from RL.agents.embedding_agent import EmbeddingAgent
from RL.agents.pushfold_agent import PushFoldAgent

from RL.agents.policy_head import PushFoldPolicy

sb_model_path = "models/saved/policies/push_fold_policy_sb.keras"
bb_model_path = "models/saved/policies/push_fold_policy_bb.keras"

def run_random_episodes(num_episodes=3, load_model=False):
    env = PushFoldEnv({
        "sb": 0.5,
        "bb": 1.0,
        "stack_bb": 10,
        "allow_variable_stack": False
    })

    if load_model == True:
        sb_policy = tf.keras.models.load_model(
            sb_model_path,
            custom_objects={"PushFoldPolicy": PushFoldPolicy},
            compile=False
        )
        bb_policy = tf.keras.models.load_model(
            bb_model_path,
            custom_objects={"PushFoldPolicy": PushFoldPolicy},
            compile=False
        )
    else:
        sb_policy = PushFoldPolicy(embedding_dim=32)
        bb_policy = PushFoldPolicy(embedding_dim=32)

    # agent = RandomAgent()
    # agent = AllInAgent()
    # agent = SBAlwaysFoldAgent()
    # agent = BBAlwaysFoldAgent()
    # agent = EmbeddingAgent()

    # policy = PushFoldPolicy(embedding_dim=encoder.output_shape[-1])
    sb_agent = PushFoldAgent(sb_policy, training=True)
    bb_agent = PushFoldAgent(bb_policy, training=True)

    total_reward_sb = 0

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode+1} ---")
        obs = env.reset()
        done = False

        print("Obs: ", obs)

        # ---------- SB acts ----------
        sb_obs = obs["embedding"]
        sb_action = sb_agent.act(obs)

        obs, reward, done, info = env.step(sb_action)
        print("Obs: ", obs)

        # ---------- BB may act ----------
        bb_obs = None
        bb_action = None

        if not done:
            bb_obs = obs["embedding"]
            bb_action = bb_agent.act(obs)
            obs, reward, done, info = env.step(bb_action)
            print("Obs: ", obs)

        # ---------- terminal reward ----------
        # reward is SB payoff
        sb_reward = reward

        if bb_action is not None:
            if bb_action == 0:  # BB FOLDS
                bb_reward = -1.0
            else:               # BB CALLS
                bb_reward = -reward
        else:
            bb_reward = None

        total_reward_sb += sb_reward  

    EV = total_reward_sb / num_episodes
    print("EV: ", EV)

if __name__ == "__main__":
    run_random_episodes(1, load_model=True)
