import itertools
import numpy as np
import tensorflow as tf

from RL.env.push_fold_env import PushFoldEnv
from RL.agents.policy_head import PushFoldPolicy

RANKS = "AKQJT98765432"
SUITS = "shdc"

MODEL_PATH = "models/saved/policies/push_fold_policy.keras"

# ---------- Utilities ----------

def card(rank, suit):
    return rank + suit

def all_cards():
    return [r + s for r in RANKS for s in SUITS]

def hand_combos(rank1, rank2, suited):
    combos = []
    if rank1 == rank2:
        # pairs
        for s1, s2 in itertools.combinations(SUITS, 2):
            combos.append([card(rank1, s1), card(rank1, s2)])
    else:
        if suited:
            for s in SUITS:
                combos.append([card(rank1, s), card(rank2, s)])
        else:
            for s1 in SUITS:
                for s2 in SUITS:
                    if s1 != s2:
                        combos.append([card(rank1, s1), card(rank2, s2)])
    return combos

def select_action(policy, obs):
    emb = obs["embedding"]
    logits = policy(tf.convert_to_tensor([emb], dtype=tf.float32))
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    return int(np.random.choice(len(probs), p=probs))

# ---------- Core evaluation ----------

def evaluate_hand_ev(hand, position, num_rollouts=500):
    env = PushFoldEnv({
        "sb": 0.5,
        "bb": 1.0,
        "stack_bb": 10,
        "allow_variable_stack": False
    })

    policy = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"PushFoldPolicy": PushFoldPolicy},
        compile=False
    )

    total_ev = 0.0

    for _ in range(num_rollouts):
        obs = env.reset()

        # Force the hand
        env.hands[position] = hand.copy()

        done = False
        while not done:
            action = select_action(policy, obs)
            obs, reward, done, info = env.step(action)

        # Compute EV from perspective of `position`
        if info["winner"] == position:
            ev = env.pot - env.committed[position]
        else:
            ev = -env.committed[position]

        total_ev += ev

    return total_ev / num_rollouts

# ---------- Grid generation ----------

def generate_ev_grid(position, num_rollouts=500):
    grid = np.zeros((13, 13))

    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i == j:
                combos = hand_combos(r1, r2, suited=False)
            elif i < j:
                combos = hand_combos(r1, r2, suited=True)
            else:
                combos = hand_combos(r1, r2, suited=False)

            evs = []
            for hand in combos:
                evs.append(evaluate_hand_ev(hand, position, num_rollouts))

            grid[i, j] = np.mean(evs)

            label = f"{r1}{r2}{'s' if i < j else 'o' if i > j else ''}"
            print(f"{label:4s} | EV: {grid[i,j]:+.3f}")

    return grid

# ---------- Entry point ----------

if __name__ == "__main__":
    print("\n=== SB EV Grid ===")
    sb_grid = generate_ev_grid(position=0, num_rollouts=200)

    print("\n=== BB EV Grid ===")
    bb_grid = generate_ev_grid(position=1, num_rollouts=200)

    np.save("sb_ev_grid.npy", sb_grid)
    np.save("bb_ev_grid.npy", bb_grid)
