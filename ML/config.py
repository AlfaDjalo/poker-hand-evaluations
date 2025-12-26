import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from ML.data.load_db import open_db
from dataclasses import dataclass

def get_config():
    db = open_db()

    base_dir = os.path.dirname(os.path.dirname(__file__))

    config = {
        # ---- General ----
        # "mode": "hand_category",
        "mode": "grid_value",
        # "mode": "alternating",
        # "mode": "embedding_value",
        "load_encoder_model": True,
        "load_head_model": True,
        "save_model": True,

        # ---- Database ----
        "db": db,
        "db_batch_size": 32000, #32000,
        "model_batch_size": 64, #1024,

        # ---- Model ----
        # "use_shared_encoder": True,
        "input_shape": (14, 4, 2),
        "encoder_input_shape": (14, 4, 1),
        # "input_shape": (13, 4, 2),
        # "input_shape_encoder": (13, 4, 1),
        # "filters": (32, 48, 64),
        # "kernel_size": 2,
        "embedding_dim": 32,
        # "hand_embedding_dim": 32,
        # "board_embedding_dim": 32,
        # "combined_embedding_dim": 32,
        # "lr": 1e-3,
        # "lr_absolute_value": 1e-4,
        # "lr_pairwise": 1e-4,
        # "loss_weights": [0.2, 0.2, 0.6],
        # "activation": "sigmoid",
        # "use_equivariance": False, #True,
        # "embedding_value_head_units": [64, 32],

        # ---- Training ----
        "epochs": 3,
        "steps_per_epoch": 500,
        # "callbacks": [],
        "mix_ratio": [0.45, 0.45, 0.1],
        "category_class_weights": [20.0, 12.0, 7.0, 5.0, 4.0, 1.0, 0.6, 0.3, 0.2],
        # "category_class_weights": [0.2, 0.3, 0.6, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0],

        # ---- Callback hyperparameters ----
        "reduce_lr": {
            "monitor": "loss", #"val_loss",
            "factor": 0.5,
            "patience": 5,
            "min_delta": 0.0, # added
            "min_lr": 1e-6
        },
        "early_stopping": {
            "monitor": "loss", #"val_loss",
            "patience": 15,
            "min_delta": 0.01, # added
            "restore_best_weights": True
        },
        "checkpoint": {
            "save_best_only": True
        },

        # ---- Paths ----
        "save_directory":                   os.path.join(base_dir, "models", "saved"),
        "embeddings_directory":             os.path.join(base_dir, "ML", "models", "embeddings"),
        
        # "hand_encoder_filename":            "hand_encoder_model.keras",
        # "board_encoder_filename":           "board_encoder_model.keras",
        # "combined_encoder_filename":        "combined_encoder_model.keras",
        # "shared_encoder_filename":          "shared_encoder_model.keras",
        "encoder_filename":                 "encoder_model.keras",
        
        # "absolute_model_filename":          "poker_value_model.keras",
        # "embedding_value_model_filename":   "poker_embedding_value_model.keras",
        # "category_model_filename":          "poker_category_model.keras",
        # "pairwise_model_filename":          "poker_pairwise_model.keras",
        
        "log_dir":                          os.path.join(base_dir, "logs")
    }

    return config

def summarize_config(config):
    print("==== CONFIG SUMMARY ====")
    for k, v in config.items():
        print(f"{k:20}: {v}")
    print("========================")

# @dataclass
# class Config:
#     mode: str = "absolute_value"
#     db_batch_size: int = 32000
#     model_batch_size: int = 64
#     embedding_dim: int = 32
#     lr: float = 1e-3
#     epochs: int = 100
#     steps_per_epoch: int = 500
#     loss_weights: tuple = (0.3, 0.3, 0.4)
#     activation: str = "sigmoid"
#     load_model: bool = True
#     save_model: bool = True
#     save_path: str = os.path.join(base_dir, "models", "saved", "poker_value_model.keras")