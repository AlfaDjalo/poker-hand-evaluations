import os, sys
import tensorflow as tf
# import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from ML.data import *
# from ML.data.generators import ValueGenerator, PairwiseComparisonGenerator, AlternatingGenerator, HandCategoryGenerator
from training.evaluation import evaluate_value, evaluate_pairwise_comparison, evaluate_hand_category
from ML.models import *

def _hand_category_config():
    return {
        "save_file": "hand_category_model.keras",
        "input_processor": SingleHandProcessor,
        "head_model": {
            "combined": CombinedInputHandCategoryHead,
            "separate": SeparateInputHandCategoryHead
        },
        "input_shape": (14, 4, 2),
        "generator": HandCategoryGenerator,
        # "default_epochs": 30,
        "default_epochs": 5,
        # Use a factory to construct a weighted categorical loss using weights from runtime config
        "loss_function": lambda cfg: WeightedCategoricalCrossentropy(cfg["category_class_weights"]),
        "loss_weights": 1.0,
        "metrics": [tf.keras.metrics.CategoricalAccuracy()],
        # "learning_rate": 1e-4,
        # "learning_rate": 1e-3,
        "evaluation_function": evaluate_hand_category,
        "evaluation_function_kwargs": {
            # "full_confusion": True
            "full_confusion": False
        }
    }

def _value_config():
    return {
        "save_file": "value_model.keras",
        "input_processor": SingleHandProcessor,
        "head_model": {
            "combined": CombinedInputValueHead,
            "separate": SeparateInputValueHead
        },
        "input_shape": (14, 4, 2),
        "generator": ValueGenerator,
        "default_epochs": 10,
        "loss_function": "mse",
        "loss_weights": 1.0,
        "metrics": ["mae"],
        # "learning_rate": 1e-3,
        "evaluation_function": evaluate_value,
    }

def _pairwise_comparison_config():
    return {
        "save_file": "pairwise_model.keras",
        "build_function": build_pairwise_comparison_model,
        "generator": PairwiseComparisonGenerator,
        "output_adapter": {
            "type": "repeat",
            "count": 3,
        },
         "evaluation_function": evaluate_pairwise_comparison,
    }


def _alternating_config():
    return {
        "save_file": "pairwise_model.keras",
        "input_processor": PairProcessor,
        "head_model": {
            "combined": CombinedInputPairwiseComparisonHead,
            "separate": SeparateInputPairwiseComparisonHead
        },
        "inputs": [
            {"name": "input_A", "shape": (14, 4, 2)},
            {"name": "input_B", "shape": (14, 4, 2)},
        ],
        "generator": AlternatingGenerator,
        "default_epochs": 50,
        "loss_function": [tf.keras.losses.BinaryCrossentropy(from_logits=False)] * 3,
        "loss_weights": [0.3, 0.3, 0.4],
        # "learning_rate": 1e-3,
        "evaluation_function": evaluate_pairwise_comparison,
    }

_CONFIG_FACTORIES = {
    "hand_category": _hand_category_config,
    "value": _value_config,
    "pairwise_comparison": _pairwise_comparison_config,
    "alternating": _alternating_config,
}

def get_mode_config(mode: str) -> dict:
    try:
        return _CONFIG_FACTORIES[mode]()
    except KeyError:
        valid = ", ".join(_CONFIG_FACTORIES.keys())
        raise ValueError(f"Unknown training mode '{mode}'. Valid modes: {valid}")

