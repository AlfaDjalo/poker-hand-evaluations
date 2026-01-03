from RL.agents.base_agent import BaseAgent
import random

FOLD = 0
ALL_IN = 1

SB = 0
BB = 1

class SBAlwaysFoldAgent(BaseAgent):
    def act(self, observation):
        if observation["position"] == SB:
            return FOLD
        else:
            return ALL_IN