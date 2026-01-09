# RL/analysis/policy_hand_grids.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path

from ML.training.trainer import get_custom_objects
from RL.agents.policy_head import PushFoldPolicy
from RL.agents.pushfold_agent import PushFoldAgent
from RL.env.push_fold_env import cards_to_tensor

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENCODER_PATH = PROJECT_ROOT / "models" / "saved" / "encoder_model.keras"
SB_POLICY_PATH = PROJECT_ROOT / "models" / "saved" / "policies" / "push_fold_policy_sb.keras"
BB_POLICY_PATH = PROJECT_ROOT / "models" / "saved" / "policies" / "push_fold_policy_bb.keras"
OUTPUT_DIR = PROJECT_ROOT / "RL" / "analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Constants ----------
RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
SUITS = ["s", "h", "d", "c"]

ACTION_ALLIN = 1  # index of ALLIN/CALL in logits

# NEW: Stack normalization constants - MUST match training config
STACK_MIN = 2.0
STACK_MAX = 20.0


# ---------- Load models ----------
encoder = tf.keras.models.load_model(
    ENCODER_PATH,
    custom_objects=get_custom_objects(),
    compile=False
)

sb_policy = tf.keras.models.load_model(
    SB_POLICY_PATH,
    custom_objects={"PushFoldPolicy": PushFoldPolicy},
    compile=False
)

bb_policy = tf.keras.models.load_model(
    BB_POLICY_PATH,
    custom_objects={"PushFoldPolicy": PushFoldPolicy},
    compile=False
)

# ---------- Create Agents ----------
sb_agent = PushFoldAgent(
    policy=sb_policy,
    training=False
)

bb_agent = PushFoldAgent(
    policy=bb_policy,
    training=False
)

# ---------- Hand utilities ----------
def all_starting_hand_combos(rank1, rank2, suited):
    cards = []
    if rank1 == rank2:
        # pocket pair: choose 2 suits
        for s1, s2 in combinations(SUITS, 2):
            cards.append([rank1 + s1, rank2 + s2])
    else:
        if suited:
            for s in SUITS:
                cards.append([rank1 + s, rank2 + s])
        else:
            for s1 in SUITS:
                for s2 in SUITS:
                    if s1 != s2:
                        cards.append([rank1 + s1, rank2 + s2])
    return cards


# UPDATED: Add stack_bb parameter
def policy_prob_for_hand(hand, position, stack_bb, mode="probs"):
    # Get card embedding
    card_embedding = encoder(
        tf.expand_dims(cards_to_tensor(hand, "hand"), axis=0),
        training=False
    )[0].numpy().flatten()
    
    # NEW: Normalize stack and append to embedding
    normalized_stack = (stack_bb - STACK_MIN) / (STACK_MAX - STACK_MIN)
    normalized_stack = np.clip(normalized_stack, 0.0, 1.0)
    
    # Combine card embedding with stack
    full_embedding = np.concatenate([card_embedding, [normalized_stack]])
    
    obs = {
        "embedding": full_embedding
    }

    if position == 0:
        agent = sb_agent
    else:
        agent = bb_agent

    if mode == "probs":
        probs = agent.policy_probs(obs["embedding"])[0]
        return_val = float(probs[ACTION_ALLIN])
    else:
        return_val = agent.policy_values(obs["embedding"])[0]

    return return_val


# UPDATED: Add stack_bb parameter
def generate_policy_grid(position, stack_bb, mode="probs"):
    grid = np.zeros((13, 13), dtype=np.float32)

    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i == j:
                combos = all_starting_hand_combos(r1, r2, suited=False)
            elif i < j:
                combos = all_starting_hand_combos(r1, r2, suited=True)
            else:
                combos = all_starting_hand_combos(r1, r2, suited=False)

            probs = [policy_prob_for_hand(hand, position, stack_bb, mode) for hand in combos]
            grid[i, j] = np.mean(probs)

    return grid


# ---------- Plotting ----------
def plot_heatmap(grid, title, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Action Probability")
    plt.xticks(range(13), RANKS)
    plt.yticks(range(13), RANKS)
    plt.xlabel("Second Card")
    plt.ylabel("First Card")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


# ---------- Main ----------
if __name__ == "__main__":
    # stack_size = 2

    for stack_size in range(5, 20, 5):
        print(f"Generating SB push grid at {stack_size}bb...")
        sb_grid = generate_policy_grid(position=0, stack_bb=stack_size, mode="probs")
        plot_heatmap(
            sb_grid,
            f"SB Push Probability at {stack_size}bb",
            f"sb_push_policy_heatmap_{stack_size}bb.png"
        )

        print(f"Generating BB call grid at {stack_size}bb...")
        bb_grid = generate_policy_grid(position=1, stack_bb=stack_size, mode="probs")
        plot_heatmap(
            bb_grid,
            f"BB Call Probability at {stack_size}bb",
            f"bb_call_policy_heatmap_{stack_size}bb.png"
        )

    print(f"Heatmaps saved to RL/analysis/")