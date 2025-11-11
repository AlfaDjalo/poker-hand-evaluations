import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from db_plo import DB_PLO, open_db
from dataclasses import dataclass

def get_config():
    db = open_db()

    base_dir = os.path.dirname(os.path.dirname(__file__))

    config = {
        # ---- General ----
        "mode": "absolute_value",
        "load_model": True,
        "save_model": True,

        # ---- Database ----
        "db": db,
        "db_batch_size": 32000,
        "model_batch_size": 64,

        # ---- Model ----
        "embedding_dim": 32,
        "lr": 1e-3,
        "loss_weights": [0.3, 0.3, 0.4],
        "activation": "sigmoid",

        # --- Training ----
        "epochs": 10, #100,
        "steps_per_epoch": 100, #500,
        "callbacks": [],

        # ---- Paths ----
        "save_path": os.path.join(base_dir, "models", "saved", "poker_value_model.keras"),
        "log_dir": os.path.join(base_dir, "logs")
    }

    return config

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