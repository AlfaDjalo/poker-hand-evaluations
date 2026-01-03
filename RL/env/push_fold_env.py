import random
import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional
from RL.evaluators.treys_eval import evaluate_hand
from ML.training.trainer import get_custom_objects

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
encoder_path = PROJECT_ROOT / "models" / "saved" / "encoder_model.keras"

class PushFoldEnv:
    ACTION_FOLD = 0
    ACTION_ALLIN = 1

    POS_SB = 0
    POS_BB = 1

    def __init__(self, config):
        self.sb_bb = config.get("sb", 0.5)
        self.bb_bb = config.get("bb", 1.0)
        self.stack_bb = config.get("stack_bb", 10)  # fixed stack for now
        self.allow_variable_stack = config.get("allow_variable_stack", False)

        self.rng = random.Random()
        self.done = True  # enforce reset before step

        # Internal state placeholders
        self.deck = []
        self.stacks = [0.0, 0.0]
        self.pot = 0.0
        self.hands = [None, None]
        self.active_player = None
        self.terminal_reason = None
        self.winner = None
        self.showdown_occurred = False

        self.encoder = tf.keras.models.load_model(encoder_path, custom_objects=get_custom_objects(), compile=False)
        self.encoder.summary()
        self.embedding_mode = "hand"

    def _get_embedding(self, hand_cards):
        tensor_input = cards_to_tensor(hand_cards, self.embedding_mode)
        tensor_batch = tf.expand_dims(tensor_input, axis=0)
        embedding = self.encoder(tensor_batch, training=False)[0].numpy()
        return embedding.flatten()

    def _get_obs(self):
        obs = {
           "hand": self.hands[self.active_player].copy(),
           "stack_bb": self.stacks[self.active_player],
           "pot_bb": self.pot,
           "sb_bb": self.sb_bb,
           "bb_bb": self.bb_bb,
           "position": self.active_player,
           "embedding": self._get_embedding(self.hands[self.active_player]),
        }
        return obs

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        else:
            self.rng.seed()

        # Setup stacks
        if self.allow_variable_stack and isinstance(self.stack_bb, tuple):
            self.stacks = [
                self.rng.uniform(self.stack_bb[0], self.stack_bb[1]),
                self.rng.uniform(self.stack_bb[0], self.stack_bb[1]),
            ]
        else:
            self.stacks = [self.stack_bb, self.stack_bb]

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

        return self._get_obs()

    def step(self, action):
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
        if action == self.ACTION_FOLD:
            # P1 folds immediately
            reward = -self.sb_bb
            self.done = True
            self.terminal_reason = "P1_fold"
            self.winner = self.POS_BB
            info = self._make_info()

            return None, reward, self.done, info

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
            return self._get_obs(), reward, self.done, info

    def _step_p2(self, action):
        if action == self.ACTION_FOLD:
            # P2 folds after P1 shove
            # When player folds:
            reward = -self.committed[self.active_player]
            self.done = True
            self.terminal_reason = "P2_fold"
            self.winner = self.POS_SB
            info = self._make_info()

            return None, reward, self.done, info

        elif action == self.ACTION_ALLIN:
            # P2 calls all-in
            call_amount = self.stacks[self.POS_BB]
            self.pot += call_amount
            self.stacks[self.POS_BB] = 0.0

            # Showdown needed
            self.done = True
            self.terminal_reason = "showdown"
            self.showdown_occurred = True

            reward = self._resolve_showdown()
            info = self._make_info()

            return None, reward, self.done, info

    def _resolve_showdown(self):
        # Placeholder showdown logic:
        # Randomly pick a winner (0 or 1)
        board = [self.deck.pop() for _ in range(5)]

        SB_hand_rank = evaluate_hand(self.hands[self.POS_SB], board)
        BB_hand_rank = evaluate_hand(self.hands[self.POS_BB], board)

        self.winner = self.POS_SB if SB_hand_rank < BB_hand_rank else self.POS_BB

        # print("SB Hand: ", self.hands[self.POS_SB], board, " Rank: ", SB_hand_rank)
        # print("BB Hand: ", self.hands[self.POS_BB], board, " Rank: ", BB_hand_rank)

        # winner = self.rng.choice([self.POS_SB, self.POS_BB])
        # self.winner = winner

        # Acting player is P2 here, so reward is from P2 perspective:
        # +pot if P2 wins, -call_amount if P2 loses
        call_amount = self.pot / 2  # since pot = sum of two all-ins + blinds

        if self.active_player == self.winner:
            reward = self.pot - self.committed[self.active_player]
        else:
            reward = -self.committed[self.active_player]

        return reward

    # def _get_obs(self):
    #     obs = {
    #         "hand": self.hands[self.active_player].copy(),
    #         "stack_bb": self.stacks[self.active_player],
    #         "pot_bb": self.pot,
    #         "sb_bb": self.sb_bb,
    #         "bb_bb": self.bb_bb,
    #         "position": self.active_player,
    #     }
    #     return obs

    def _make_info(self):
        return {
            "terminal_reason": self.terminal_reason,
            "winner": self.winner,
            "showdown": self.showdown_occurred,
        }

    def _create_deck(self):
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

