from RL.agents.base_agent import BaseAgent
import random

FOLD = 0
ALL_IN = 1

SB = 0
BB = 1

class RandomAgent(BaseAgent):
    def act(self, observation):
        if observation["position"] == SB:
            return random.choice([FOLD, ALL_IN])
        else:
            return random.choice([FOLD, ALL_IN])
