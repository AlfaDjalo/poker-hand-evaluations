from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Agents receive an observation from the environment and must
    return a valid action for the current player.
    """

    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        
    @abstractmethod
    def act(self, observation):
        """
        Choose an action given the current observation.
        
        Args:
            observation (dict): Environment observation for the current player.
            
        Returns:
            action: An environment-compatible action (e.g. int or enum).
        """

    def reset(self):
        """
        Reset any per-episode state.
        Stateless agents can ignore this.
        """
        pass