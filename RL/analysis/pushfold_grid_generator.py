# RL/analysis/pushfold_grid_generator.py

import numpy as np
import tensorflow as tf
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

# ---------- Constants ----------
RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
SUITS = ["s", "h", "d", "c"]

ACTION_ALLIN = 1  # index of ALLIN/CALL in logits

# Stack normalization constants - MUST match training config
STACK_MIN = 2.0
STACK_MAX = 20.0

# ---------- Load models (cache these) ----------
_models_loaded = False
_encoder = None
_sb_policy = None
_bb_policy = None
_sb_agent = None
_bb_agent = None

def load_models():
    """Load models once and cache them"""
    global _models_loaded, _encoder, _sb_policy, _bb_policy, _sb_agent, _bb_agent
    
    if _models_loaded:
        return
    
    _encoder = tf.keras.models.load_model(
        ENCODER_PATH,
        custom_objects=get_custom_objects(),
        compile=False
    )
    
    _sb_policy = tf.keras.models.load_model(
        SB_POLICY_PATH,
        custom_objects={"PushFoldPolicy": PushFoldPolicy},
        compile=False
    )
    
    _bb_policy = tf.keras.models.load_model(
        BB_POLICY_PATH,
        custom_objects={"PushFoldPolicy": PushFoldPolicy},
        compile=False
    )
    
    _sb_agent = PushFoldAgent(policy=_sb_policy, training=False)
    _bb_agent = PushFoldAgent(policy=_bb_policy, training=False)
    
    _models_loaded = True


def all_starting_hand_combos(rank1, rank2, suited):
    """Generate all card combinations for a given hand type"""
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


def get_embedding_with_stack(hand, stack_bb):
    """Create embedding that includes both hand and stack information"""
    # Get card embedding
    tensor_input = cards_to_tensor(hand, mode="hand")
    tensor_batch = tf.expand_dims(tensor_input, axis=0)
    card_embedding = _encoder(tensor_batch, training=False)[0].numpy().flatten()
    
    # Normalize stack to [0, 1]
    normalized_stack = (stack_bb - STACK_MIN) / (STACK_MAX - STACK_MIN)
    normalized_stack = np.clip(normalized_stack, 0.0, 1.0)
    
    # Combine
    full_embedding = np.concatenate([card_embedding, [normalized_stack]])
    
    return full_embedding

def policy_eval_for_hand(hand, position, stack_bb, mode="probs"):
    embedding = get_embedding_with_stack(hand, stack_bb)
    
    if position in ('sb', 0):
        agent = _sb_agent
    else:
        agent = _bb_agent

    if mode == "probs":
        probs = agent.policy_probs(embedding)[0]
        return float(probs[ACTION_ALLIN])

    elif mode == "values":
        value = agent.policy_values(embedding)
        return float(value.numpy()[0])

    else:
        raise ValueError(f"Unknown mode: {mode}")

def policy_prob_for_hand(hand, position, stack_bb):
    """Get push/call probability for a specific hand at a specific stack depth"""
    embedding = get_embedding_with_stack(hand, stack_bb)
    
    if position in ('sb', 0):
        agent = _sb_agent
    else:
        agent = _bb_agent

    # agent = _sb_agent if position == 'sb' else _bb_agent
    
    # Get action probabilities
    # probs = agent.action_probs({"embedding": embedding})
    probs = agent.policy_probs(embedding)[0]

    return float(probs[ACTION_ALLIN])

def generate_pushfold_grid_data(stack_bb, position):
    load_models()
    
    prob_grid = []
    value_grid = []
    combos = {}

    for i, r1 in enumerate(RANKS):
        prob_row = []
        value_row = []
        for j, r2 in enumerate(RANKS):
            if i == j:
                hand_notation = r1 + r2
                suited = False
            elif i < j:
                hand_notation = r1 + r2 + 's'
                suited = True
            else:
                hand_notation = r2 + r1 + 'o'
                suited = False
            
            if i == j:
                hand_combos = all_starting_hand_combos(r1, r2, suited=False)
            elif i < j:
                hand_combos = all_starting_hand_combos(r1, r2, suited=True)
            else:
                hand_combos = all_starting_hand_combos(r1, r2, suited=False)
            
            combo_details = []
            prob_values = []
            value_values = []
            
            for combo in hand_combos:
                prob = policy_eval_for_hand(combo, position, stack_bb, mode="probs")
                val = policy_eval_for_hand(combo, position, stack_bb, mode="values")
                combo_details.append({
                    "cards": combo,
                    "probability": prob,
                    "value": val
                })
                prob_values.append(prob)
                value_values.append(val)
            
            avg_prob = np.mean(prob_values)
            avg_val = np.mean(value_values)
            
            prob_row.append(avg_prob)
            value_row.append(avg_val)
            
            combos[hand_notation] = {
                "combos": combo_details,
                "average_prob": avg_prob,
                "average_value": avg_val,
                "count": len(combo_details)
            }
        
        prob_grid.append(prob_row)
        value_grid.append(value_row)
    
    return {
        "prob_grid": prob_grid,
        "value_grid": value_grid,
        "combos": combos,
        "stack_bb": stack_bb,
        "position": position,
    }

def generate_pushfold_grid_data_old(stack_bb, position, mode='probs'):
    """
    Generate complete grid data including combo breakdowns
    
    Args:
        stack_bb: Stack size in big blinds
        position: 'sb' or 'bb'
        mode: 'probs' or 'values' (currently only probs implemented)
    
    Returns:
        dict with:
            - grid: 13x13 array of average values
            - combos: dict mapping hand notation to combo details
            - stack_bb: the stack size used
            - position: the position
    """
    load_models()
    
    grid = []
    combos = {}
    
    for i, r1 in enumerate(RANKS):
        row = []
        for j, r2 in enumerate(RANKS):
            # Determine hand notation and if suited
            if i == j:
                # Pocket pair
                hand_notation = r1 + r2
                suited = False
            elif i < j:
                # Suited (above diagonal)
                hand_notation = r1 + r2 + 's'
                suited = True
            else:
                # Offsuit (below diagonal)
                hand_notation = r2 + r1 + 'o'
                suited = False
            
            # Get all combos for this hand
            if i == j:
                hand_combos = all_starting_hand_combos(r1, r2, suited=False)
            elif i < j:
                hand_combos = all_starting_hand_combos(r1, r2, suited=True)
            else:
                hand_combos = all_starting_hand_combos(r1, r2, suited=False)

            # hand_combos = all_starting_hand_combos(
            #     r1 if i == j or i < j else r2,
            #     r2 if i == j or i < j else r1,
            #     suited
            # )
            
            # Calculate probability for each combo
            combo_details = []
            for combo in hand_combos:
                prob = policy_eval_for_hand(combo, position, stack_bb, mode)
                # prob = policy_prob_for_hand(combo, position, stack_bb)
                combo_details.append({
                    'cards': combo,
                    'probability': prob
                })
            
            # Average probability across all combos
            avg_prob = np.mean([c['probability'] for c in combo_details])
            
            row.append(avg_prob)
            combos[hand_notation] = {
                'combos': combo_details,
                'average': avg_prob,
                'count': len(combo_details)
            }
        
        grid.append(row)
    
    return {
        'grid': grid,
        'combos': combos,
        'stack_bb': stack_bb,
        'position': position,
        'mode': mode
    }