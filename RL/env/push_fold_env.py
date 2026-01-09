import random
import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional

# import os, sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ML")))


from RL.evaluators.treys_eval import evaluate_hand

from ML.training.trainer import get_custom_objects

# from bindings.equity_wrapper import compute_equity as compute_calc
from bindings.holdem_wrapper import evaluate_showdown as cpp_evaluate_showdown

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
encoder_path = PROJECT_ROOT / "models" / "saved" / "encoder_model.keras"

class PushFoldEnv:
    """
    Push-fold Environment.
    """
    ACTION_FOLD = 0
    ACTION_ALLIN = 1

    POS_SB = 0
    POS_BB = 1

    def __init__(self, config: Dict):
        """
        Initialization of push fold class
        
        Parameters
        ----------
        config
            Environment configuration parameters, including:
            - sb (float): Small blind in bb
            - bb (float): Big blind in bb
        - stack_bb (float or tuple): If float, fixed stack. If tuple (min, max), variable stacks
        - Both players always get the same stack size
        """
        self.sb_bb = config.get("sb", 0.5)
        self.bb_bb = config.get("bb", 1.0)

        stack_config = config.get("stack_bb", 10)

        if isinstance(stack_config, tuple) and len(stack_config) == 2:
            self.stack_bb_min, self.stack_bb_max = stack_config
            self.allow_variable_stack = True
        else:
            self.stack_bb_min = float(stack_config)
            self.stack_bb_max = float(stack_config)
            self.allow_variable_stack = False

        # self.stack_bb = config.get("stack_bb", 10)  # fixed stack for now
        # self.allow_variable_stack = config.get("allow_variable_stack", False)

        self.rng = random.Random()
        self.done = True  # enforce reset before step

        # Internal state placeholders
        self.deck = []
        self.stacks = [0.0, 0.0]
        self.initial_stack = 0.0
        self.committed = [0.0, 0.0]
        self.pot = 0.0
        self.hands = [None, None]
        self.active_player = None
        self.terminal_reason = None
        self.winner = None
        self.showdown_occurred = False

        self.encoder = tf.keras.models.load_model(encoder_path, custom_objects=get_custom_objects(), compile=False)

    def _get_embedding(self, hand_cards, stack_bb):
        """
        Create embedding combining hand cards and stack size
        
        Parameters
        ----------
        hand_cards : list
            The player's hole cards
        stack_bb : float
            The starting stack size in BB (before blinds)
            
        Returns
        -------
        numpy.ndarray
            Combined embedding: [card_embedding, normalized_stack]
            Stack is normalized to [0, 1] where 0 = min_stack, 1 = max_stack
        """      
        tensor_input = cards_to_tensor(hand_cards, mode="hand")
        tensor_batch = tf.expand_dims(tensor_input, axis=0)
        card_embedding = self.encoder(tensor_batch, training=False)[0].numpy().flatten()

        normalized_stack = (stack_bb - self.stack_bb_min) / (self.stack_bb_max - self.stack_bb_min)
        normalized_stack = np.clip(normalized_stack, 0.0, 1.0)

        full_embedding = np.concatenate([card_embedding, [normalized_stack]])
        
        return full_embedding


    def _get_obs(self):
        """
        Get observation for the active player
        """
        obs = {
            "hand": self.hands[self.active_player].copy(),
            "stack_bb": self.stacks[self.active_player],
            "initial_stack_bb": self.initial_stack,
            "pot_bb": self.pot,
            "position": self.active_player,
            "embedding": self._get_embedding(self.hands[self.active_player], self.initial_stack),
        }
        return obs

    def reset(self, seed=None):
        """
        Reset episode, resetting all environment variables

        Parameters
        ----------
        seed
            Seed for random number generator

        Returns
        -------
        dict
            Output from _get_obs() function containing:
            - hand (list of str): Hand of active player
            - stack_bb (float): Stack in BB
            - pot_bb (float): Pot in BB
            - sb_bb (float): Small blind in BB
            - bb_bb (float): Big blind in BB
            - position (int): Position of active player (SB:0, BB:1)
            - embedding (list of float): Embedding of the hand
        """
        if seed is not None:
            self.rng.seed(seed)
        else:
            self.rng.seed()

        # Setup stacks
        if self.allow_variable_stack:
            stack_size = self.rng.uniform(self.stack_bb_min, self.stack_bb_max)
        else:
            stack_size = self.stack_bb_min

        self.initial_stack = stack_size  # Store for reward normalization
        self.stacks = [stack_size, stack_size]

        self.pot = 0.0
        self.done = False
        self.terminal_reason = None
        self.winner = None
        self.showdown_occurred = False
        self.active_player = self.POS_SB  # SB acts first

        # Create and shuffle deck
        self.deck = self._create_deck()
        self.rng.shuffle(self.deck)

        # Post blinds
        self.stacks[self.POS_SB] -= self.sb_bb
        self.stacks[self.POS_BB] -= self.bb_bb
        self.pot += self.sb_bb + self.bb_bb
        self.committed = [self.sb_bb, self.bb_bb]

        # Deal hands (2 cards each)
        self.hands[self.POS_SB] = [self.deck.pop(), self.deck.pop()]
        self.hands[self.POS_BB] = [self.deck.pop(), self.deck.pop()]
        # print("Hands: ", self.hands)
        return self._get_obs()

    def step(self, action):
        """
        Step in enviroment.

        Parameters
        ----------
        action
            Action of active player

        Returns
        -------
        tuple
            Output from active player step function containing:
            - obs:
            - reward:
            - done:
            - info:
        """
        if self.done:
            raise RuntimeError("step() called after episode is done; call reset() first")

        # Validate action
        if action not in (self.ACTION_FOLD, self.ACTION_ALLIN):
            raise ValueError(f"Invalid action {action}")

        if self.active_player == self.POS_SB:
            return self._step_p1(action)
        else:
            return self._step_p2(action)

    def _step_p1(self, action):
        """
        Step in enviroment for player 1 action.

        Parameters
        ----------
        action
            Action of active player

        Returns
        -------
        tuple
            Containing:
            - obs: Observation of environment from _get_obs()
            - reward: Reward to active player from episode
            - done: Episode complete flag set to True at showdown or player folding
            - info: Information regarding completion of episode
        """
        if action == self.ACTION_FOLD:
            # P1 folds immediately
            sb_reward = -self.sb_bb
            bb_reward = self.sb_bb
            self.done = True
            self.terminal_reason = "P1_fold"
            self.winner = self.POS_BB
            info = self._make_info()
            info['sb_reward'] = sb_reward
            info['bb_reward'] = bb_reward

            return None, sb_reward, self.done, info

        elif action == self.ACTION_ALLIN:
            # P1 pushes entire stack (all-in)
            # Adjust pot & stacks
            push_amount = self.stacks[self.POS_SB]
            self.pot += push_amount
            self.committed[self.POS_SB] += push_amount
            self.stacks[self.POS_SB] = 0.0

            # Now P2 acts
            self.active_player = self.POS_BB
            reward = 0.0
            info = self._make_info()
            info['sb_reward'] = 0.0  # No reward yet for SB
            info['bb_reward'] = 0.0            
            return self._get_obs(), reward, self.done, info

    def _step_p2(self, action):
        """
        Step in enviroment for player 2 action.

        Parameters
        ----------
        action
            Action of active player

        Returns
        -------
        tuple
            Containing:
            - obs: Observation of environment from _get_obs()
            - reward: Reward to active player from episode
            - done: Episode complete flag set to True at showdown or player folding
            - info: Information regarding completion of episode
        """
        if action == self.ACTION_FOLD:
            # P2 folds after P1 shove
            # When player folds:
            sb_reward = self.bb_bb  # SB wins the pot (BB's big blind)
            bb_reward = -self.bb_bb
            # reward = -self.committed[self.active_player]
            self.done = True
            self.terminal_reason = "P2_fold"
            self.winner = self.POS_SB
            info = self._make_info()
            info['sb_reward'] = sb_reward
            info['bb_reward'] = bb_reward

            return None, bb_reward, self.done, info

        elif action == self.ACTION_ALLIN:
            # P2 calls all-in
            call_amount = self.stacks[self.POS_BB]
            self.pot += call_amount
            self.committed[self.POS_BB] += call_amount
            self.stacks[self.POS_BB] = 0.0

            # Showdown needed
            self.done = True
            self.terminal_reason = "showdown"
            self.showdown_occurred = True

            sb_reward, bb_reward = self._resolve_showdown()
            info = self._make_info()
            info['sb_reward'] = sb_reward
            info['bb_reward'] = bb_reward

            return None, bb_reward, self.done, info

    def _resolve_showdown(self):
        """
        Resolve showdown when SB action = Push and BB action = Call
        """
        board = [self.deck.pop() for _ in range(5)]
        
        # Use C++ evaluator - returns 0 if SB wins, 1 if BB wins
        self.winner = cpp_evaluate_showdown(
            self.hands[self.POS_SB],
            self.hands[self.POS_BB],
            board,
            debug=False
        )

        if self.winner == self.POS_SB:
            sb_reward = self.pot - self.committed[self.POS_SB]
            bb_reward = -self.committed[self.POS_BB]
        else:
            sb_reward = -self.committed[self.POS_SB]
            bb_reward = self.pot - self.committed[self.POS_BB]

        self.stacks[self.POS_SB] = 0        
        self.stacks[self.POS_BB] = 0        
        self.stacks[self.winner] = self.pot

        return sb_reward, bb_reward

    def _make_info(self):
        """
        Helper function to make dictionary of information on episode termination.

        Parameters
        ----------
        nil
        
        Returns
        -------
        dict
            Containing:
            - terminal reason (str): description of reason for termination of hand
            - winner (int): Position of winner of hand SB = 0, BB = 1 
            - showdown (bool): Episode complete flag set to True at showdown or player folding
        """
        # print("terminal_reason", self.terminal_reason, " winner", self.winner)
        return {
            "terminal_reason": self.terminal_reason,
            "winner": self.winner,
            "showdown": self.showdown_occurred,
        }

    def _create_deck(self):
        """
        Helper function to create a 52 card deck.

        Parameters
        ----------
        Nil

        Returns
        -------
        list of str
            Card strings representing the 52 card deck
        """
        ranks = "23456789TJQKA"
        suits = "cdhs"  # clubs, diamonds, hearts, spades
        return [r + s for r in ranks for s in suits]


def cards_to_tensor(cards: List[str], mode: str) -> tf.Tensor:
    """
    Convert cards to a (13, 4, 2) tensor.

    Channels:
        0 = hand cards
        1 = board cards

    Modes:
        - "hand":     all cards go to hand channel
        - "board":    all cards go to board channel
        - "combined": cards[0:2] -> hand channel, cards[2:5] -> board channel

    Args:
        cards: List[str]   e.g. ["As", "Kd"] or ["As", "Kd", "7h", "2d", "Tc"]
        mode: str          "hand", "board", "combined"

    Returns:
        tf.Tensor of shape (13, 4, 2)
    """

    # ----- Rank & suit lookup -----
    rank_to_idx = {
        'A': 0, 'K': 1, 'Q': 2, 'J': 3, 'T': 4,
        '9': 5, '8': 6, '7': 7, '6': 8, '5': 9,
        '4': 10, '3': 11, '2': 12
    }
    suit_to_idx = {'s': 0, 'h': 1, 'd': 2, 'c': 3}

    # ----- Base empty grid -----
    grid = np.zeros((14, 4, 1), dtype=np.float32)

    # ----- Split cards based on mode -----
    if mode == "hand":
        hand_cards = cards
        board_cards = []

    elif mode == "board":
        hand_cards = []
        board_cards = cards

    elif mode == "combined":
        hand_cards = cards[:2]
        board_cards = cards[2:5]

    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected: hand, board, combined.")

    # ----- Fill hand channel (0) -----
    for card in hand_cards:
        if len(card) < 2:
            continue
        rank = card[0].upper()
        suit = card[-1].lower()
        if rank in rank_to_idx and suit in suit_to_idx:
            grid[rank_to_idx[rank], suit_to_idx[suit], 0] = 1.0

    # ----- Fill board channel (1) -----
    for card in board_cards:
        if len(card) < 2:
            continue
        rank = card[0].upper()
        suit = card[-1].lower()
        if rank in rank_to_idx and suit in suit_to_idx:
            grid[rank_to_idx[rank], suit_to_idx[suit], 0] = 1.0

    return tf.constant(grid, dtype=tf.float32)

