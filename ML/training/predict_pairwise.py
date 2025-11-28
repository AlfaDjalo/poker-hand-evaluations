"""Utility: predict pairwise hand1 win probs from CSV using pairwise model

CSV: columns start with hand1_ and hand2_ (e.g. hand1_card1, hand1_card2,...)
"""
import sys, os
import csv
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

def predict_pairwise_from_csv(csv_path, model_path=None, config=None):
    # Lazy import config if not supplied
    if config is None:
        try:
            from config import get_config
        except Exception:
            from ML.config import get_config
        config = get_config()

    # Determine model path
    if model_path is None:
        model_path = os.path.join(config["save_directory"], config["pairwise_model_filename"])

    # Load model (try loading via Keras with custom objects if available)
    try:
        from training.trainer import get_custom_objects
        custom_objs = get_custom_objects()
    except Exception:
        custom_objs = {}

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objs, compile=False)
    else:
        # fall back to building model from config
        try:
            from models.implementation import build_pairwise_model
            model = build_pairwise_model(config)
            print(f"⚠️ model file not found at {model_path}, built model from config instead")
        except Exception as e:
            raise RuntimeError(f"Failed to load or build pairwise model: {e}")

    # Read CSV
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return []

    # helpers to convert card strings to grids
    RANKS = 'AKQJT98765432'
    rank_to_idx = {r: i for i, r in enumerate(RANKS)}
    suit_to_idx = {'s': 0, 'S': 0, 'h': 1, 'H': 1, 'd': 2, 'D': 2, 'c': 3, 'C': 3}

    def hand_cards_from_row(row, prefix):
        cards = []
        for k, v in row.items():
            if k.startswith(prefix) and v and v.strip():
                cards.append(v.strip())
        return cards

    def cards_to_grid(cards):
        grid = np.zeros((13, 4), dtype=np.float32)
        for c in cards:
            if not c or len(c) < 2:
                continue
            rank = c[:-1].upper()
            suit = c[-1]
            if rank == '10':
                rank = 'T'
            if rank not in rank_to_idx or suit not in suit_to_idx:
                # fallback: try first char as rank and second as suit
                r = c[0].upper()
                s = c[1] if len(c) > 1 else ''
                if r in rank_to_idx and s in suit_to_idx:
                    ri = rank_to_idx[r]
                    si = suit_to_idx[s]
                else:
                    continue
            else:
                ri = rank_to_idx[rank]
                si = suit_to_idx[suit]
            grid[ri, si] = 1.0
        return grid

    # Build batch inputs
    X_A = []
    X_B = []
    for row in rows:
        cards1 = hand_cards_from_row(row, 'hand1')
        cards2 = hand_cards_from_row(row, 'hand2')

        hand1_grid = cards_to_grid(cards1)[..., None]  # (13,4,1)
        hand2_grid = cards_to_grid(cards2)[..., None]

        # top-level pairwise model expects full inputs (13,4,2): hand + board
        board_zero = np.zeros((13, 4, 1), dtype=np.float32)
        inputA = np.concatenate([hand1_grid, board_zero], axis=-1)
        inputB = np.concatenate([hand2_grid, board_zero], axis=-1)

        X_A.append(inputA)
        X_B.append(inputB)

    X_A = np.stack(X_A)
    X_B = np.stack(X_B)

    # Predict
    preds = model.predict([X_A, X_B])

    # preds may be list/tuple of 3 outputs (hand_prob, board_prob, combined_prob)
    if isinstance(preds, (list, tuple)) and len(preds) >= 1:
        hand_probs = preds[0]
    else:
        # single-output pairwise model: assume it is hand comparison
        hand_probs = preds

    hand_probs = np.asarray(hand_probs).reshape(-1).tolist()
    return hand_probs


if __name__ == '__main__':
    # Run: python -m ML.training.predict_pairwise ML\\data\\test_hands.csv
    
    # quick CLI for convenience
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='CSV file of hand pairs')
    parser.add_argument('--model', help='pairwise model path', default=None)
    args = parser.parse_args()

    probs = predict_pairwise_from_csv(args.csv, model_path=args.model)
    for p in probs:
        print(p)
