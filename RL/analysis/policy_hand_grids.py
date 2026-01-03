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


def policy_prob_for_hand(hand, position):
    obs = {
        "hand": hand,
        "position": position,
        "embedding": encoder(
            tf.expand_dims(cards_to_tensor(hand, "hand"), axis=0),
            training=False
        )[0].numpy()
    }

    if position == 0:
        probs = sb_agent.action_probs(obs)[0]
    else:
        probs = bb_agent.action_probs(obs)[0]

    return probs[ACTION_ALLIN]


# def policy_prob_for_hand(hand, position):
#     tensor = cards_to_tensor(hand, mode="hand")
#     emb = encoder(tf.expand_dims(tensor, axis=0), training=False)[0]
#     logits = policy(tf.expand_dims(emb, axis=0), training=False)
#     probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
#     prob = probs[0]
#     print("Hand: ", hand, " Prob: ", prob)
#     return prob
#     # return probs[ACTION_PUSH_CALL]


# ---------- Grid generation ----------
def generate_policy_grid(position):
    grid = np.zeros((13, 13), dtype=np.float32)

    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i == j:
                combos = all_starting_hand_combos(r1, r2, suited=False)
            elif i < j:
                combos = all_starting_hand_combos(r1, r2, suited=True)
            else:
                combos = all_starting_hand_combos(r1, r2, suited=False)

            probs = [policy_prob_for_hand(hand, position) for hand in combos]
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
    print("Generating SB push grid...")
    sb_grid = generate_policy_grid(position=0)
    plot_heatmap(
        sb_grid,
        "SB Push Probability",
        "sb_push_policy_heatmap.png"
    )

    print("Generating BB call grid...")
    bb_grid = generate_policy_grid(position=1)
    plot_heatmap(
        bb_grid,
        "BB Call Probability",
        "bb_call_policy_heatmap.png"
    )

    print("Heatmaps saved to RL/analysis/")
