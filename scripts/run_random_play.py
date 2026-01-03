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

model_path = "models/saved/policies/push_fold_policy.keras"

def run_random_episodes(num_episodes=3, load_model=False):
    env = PushFoldEnv({
        "sb": 0.5,
        "bb": 1.0,
        "stack_bb": 10,
        "allow_variable_stack": False
    })

    if load_model == True:
        policy = tf.keras.models.load_model(
            model_path,
            custom_objects={"PushFoldPolicy": PushFoldPolicy},
            compile=False
        )
    else:
        policy = PushFoldPolicy(embedding_dim=32)

    # agent = RandomAgent()
    # agent = AllInAgent()
    # agent = SBAlwaysFoldAgent()
    # agent = BBAlwaysFoldAgent()
    # agent = EmbeddingAgent()

    # policy = PushFoldPolicy(embedding_dim=encoder.output_shape[-1])
    agent = PushFoldAgent(policy, training=True)

    total_reward_sb = 0

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode+1} ---")
        obs = env.reset()
        done = False

        while not done:
            print(f"Player {obs['position']} Observation:")
            print(f"  Hand: {obs['hand']}")
            print(f"  Stack: {obs['stack_bb']:.2f} BB")
            print(f"  Pot: {obs['pot_bb']:.2f} BB")
            # print(f"  Embedding: {obs['embedding']}")

            action = agent.act(obs)
            action_str = "FOLD" if action == 0 else ("PUSH" if obs['position'] == 0 else "CALL")
            print(f"Player {obs['position']} action: {action_str}")

            obs, reward, done, info = env.step(action)

            if reward != 0:
                print(f"Reward: {reward:.2f}")
            if done:
                print(f"Terminal info: {info}")

        if info["winner"] == env.POS_SB:
            reward_sb = env.pot - env.committed[env.POS_SB]  # SB wins pot minus their investment
        else:
            reward_sb = -env.committed[env.POS_SB]           # SB lost what they invested

        total_reward_sb += reward_sb  

    EV = total_reward_sb / num_episodes
    print("EV: ", EV)

if __name__ == "__main__":
    run_random_episodes(100, load_model=True)
