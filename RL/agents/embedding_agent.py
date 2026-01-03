from RL.agents.base_agent import BaseAgent
import random
import numpy as np

FOLD = 0
ALL_IN = 1

SB = 0
BB = 1

class EmbeddingAgent:
    def act(self, obs):
        print("‖emb‖ =", np.linalg.norm(obs["embedding"]))
        return random.choice([FOLD, ALL_IN])