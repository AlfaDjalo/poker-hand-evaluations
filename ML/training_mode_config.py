
import os, sys
import tensorflow as tf
# import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from ML.data.generators import ValueGenerator, PairwiseComparisonGenerator, AlternatingGenerator, HandCategoryGenerator
from training.evaluation import evaluate_value, evaluate_pairwise_comparison, evaluate_hand_category
from ML.models.implementation import *

def get_training_mode_config(mode):
    TRAINING_MODES = {
        "hand_category": {
            "save_file": "hand_category_model.keras",
            "build_function": build_hand_category_model,
            "generator": HandCategoryGenerator,
            "generator_kwargs": {"mode_override": "mix"},
            "evaluation_function": evaluate_hand_category,
            "default_epochs": 20,
            "loss_function": [tf.keras.losses.CategoricalCrossentropy()] * 3,
            "loss_weights": [0.2, 0.2, 0.6],

            # "metrics": [...],
            # "num_outputs": 3,
            # "label_format": "list",
            # "evaluate_fn": evaluate_hand_category,
        },
        "grid_value": {
            "save_file": "grid_value_model.keras",
            "build_function": build_grid_value_model,
            "generator": ValueGenerator,
            "evaluation_function": evaluate_value,
        },
        "pairwise": {
            "save_file": "pairwise_model.keras",
            "build_function": build_pairwise_comparison_model,
            "generator": PairwiseComparisonGenerator,
            "evaluation_function": evaluate_pairwise_comparison,
        },
        "embedding_value": {
            "save_file": "embedding_value_model.keras",
            "build_function": build_embedding_value_model,
            "generator": ValueGenerator,
            "evaluation_function": evaluate_value,
        },
    }

    return TRAINING_MODES[mode]