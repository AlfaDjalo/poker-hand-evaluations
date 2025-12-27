import os, sys
import tensorflow as tf
# import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from ML.data.generators import ValueGenerator, PairwiseComparisonGenerator, AlternatingGenerator, HandCategoryGenerator
from training.evaluation import evaluate_grid_value, evaluate_value, evaluate_pairwise_comparison, evaluate_hand_category
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
        "default_epochs": 20,
        # Use a factory to construct a weighted categorical loss using weights from runtime config
        "loss_function": lambda cfg: [WeightedCategoricalCrossentropy(cfg["category_class_weights"])],
        "loss_weights": 1.0,
        "metrics": [tf.keras.metrics.CategoricalAccuracy()],
        "learning_rate": 1e-3,
        "evaluation_function": evaluate_hand_category,
        "evaluation_function_kwargs": {
            "full_confusion": True
        }
    }

def _old_hand_category_config():
    return {
        "save_file": "hand_category_model.keras",
        "build_function": build_hand_category_model,
        "generator": HandCategoryGenerator,
        "output_adapter": {
            "type": "repeat",
            "count": 3,
        },
        "default_epochs": 20,
        # Use a factory to construct a weighted categorical loss using weights from runtime config
        "loss_function": lambda cfg: [WeightedCategoricalCrossentropy(cfg["category_class_weights"])] * 3,
        "loss_weights": [0.2, 0.2, 0.6],
        "metrics": [tf.keras.metrics.CategoricalAccuracy()] * 3,
        "learning_rate": 1e-3,
        "evaluation_function": evaluate_hand_category,
        "evaluation_function_kwargs": {
            "full_confusion": True
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
        "output_adapter": {
            "type": "identity",
        },
        "loss_function": "mse",
        "loss_weights": 1.0,
        "metrics": ["mae"],
        "learning_rate": 1e-3,
        "evaluation_function": evaluate_value,
    }

def _old_value_config():
    return {
        "save_file": "value_model.keras",
        "build_function": build_value_model,
        "generator": ValueGenerator,
        "output_adapter": {
            "type": "identity",
        },
        "loss_function": "mse",
        "loss_weights": 1.0,
        "learning_rate": 1e-3,
        "metrics": ["mae"],
        "learning_rate": 1e-3,
        "evaluation_function": evaluate_value,
    }

def _grid_value_config():
    return {
        "save_file": "grid_value_model.keras",
        "build_function": build_value_model,
        # "build_function": build_grid_value_model,
        "generator": ValueGenerator,
        "output_adapter": {
            "type": "identity",
        },
        "loss_function": "mse",
        "loss_weights": 1.0,
        "learning_rate": 1e-3,
        "metrics": ["mae"],
        # "output_adapter": {
        #     "type": "repeat",
        #     "count": 3,
        # },
        # "loss_function": ["mse"] * 3,
        # "loss_weights": [0.2, 0.2, 0.6],
        # "metrics": ["mae"] * 3,
        "learning_rate": 1e-3,
        # "evaluation_function": evaluate_embedding_value,
        # "evaluation_function": evaluate_grid_value,
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
        "loss_function": [tf.keras.losses.BinaryCrossentropy(from_logits=False)] * 3,
        "loss_weights": [0.3, 0.3, 0.4],
        "learning_rate": 1e-3,
        "evaluation_function": evaluate_pairwise_comparison,
    }

def _old_alternating_config():
    return {
        "save_file": "pairwise_model.keras",
        "build_function": build_pairwise_comparison_model,
        "generator": AlternatingGenerator,
        "output_adapter": {
            "type": "repeat",
            "count": 3,
        },
        "loss_function": [tf.keras.losses.BinaryCrossentropy(from_logits=False)] * 3,
        "loss_weights": [0.3, 0.3, 0.4],
        "learning_rate": 1e-3,
        "evaluation_function": evaluate_pairwise_comparison,
    }


def _embedding_value_config():
    return {
        "save_file": "embedding_value_model.keras",
        "build_function": build_embedding_value_model,
        "generator": ValueGenerator,
        "output_adapter": {
            "type": "identity",
        },
        "loss_function": "mse",
        "loss_weights": 1.0,
        "learning_rate": 1e-3,
        "metrics": ["mae"],
        # "evaluation_function": evaluate_embedding_value,
    }

_CONFIG_FACTORIES = {
    "hand_category": _hand_category_config,
    "value": _value_config,
    "grid_value": _grid_value_config,
    "pairwise_comparison": _pairwise_comparison_config,
    "alternating": _alternating_config,
    "embedding_value": _embedding_value_config,
}

def get_mode_config(mode: str) -> dict:
    try:
        return _CONFIG_FACTORIES[mode]()
    except KeyError:
        valid = ", ".join(_CONFIG_FACTORIES.keys())
        raise ValueError(f"Unknown training mode '{mode}'. Valid modes: {valid}")

