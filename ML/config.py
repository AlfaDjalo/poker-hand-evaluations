import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from ML.data.load_db import open_db
from dataclasses import dataclass

def get_config():
    db = open_db()

    base_dir = os.path.dirname(os.path.dirname(__file__))

    config = {
        # ---- Modes ----
        "mode": "hand_category",
        # "mode": "value",
        # "mode": "alternating",

        # "submode": "combined",
        "submode": "separate",
        
        # ---- Save / Load Models ----
        "load_encoder_model": False,
        "load_head_model": False,
        # "load_head_model": True,
        # "save_model": False,

        # New: control saving of encoder vs head separately (scheduler uses these)
        "save_encoder_model": False,
        "save_head_model": False,

        # New: schedule files (scheduler will read/write here by default)
        "schedules_directory":               os.path.join(base_dir, "ML", "schedules"),
        # "schedule_file":                     os.path.join(base_dir, "ML", "schedules", "test.yaml"),
        "schedule_file":                     os.path.join(base_dir, "ML", "schedules", "all_tasks.yaml"),
        "schedule_output":                   os.path.join(base_dir, "ML", "schedules", "training_schedule_results.json"),

        # ---- Database ----
        "db": db,
        "db_batch_size": 32000, #32000,
        "model_batch_size": 64, #1024,

        # ---- Model ----
        "input_shape": (14, 4, 2),
        "encoder_input_shape": (14, 4, 1),

        "embedding_dim": 32,

        # ---- Training ----
        # "epochs": 3,
        "steps_per_epoch": 500,
        # "callbacks": [],
        "mix_ratio": [0.45, 0.45, 0.1],
        "category_class_weights": [20.0, 12.0, 7.0, 5.0, 4.0, 1.0, 0.6, 0.3, 0.2],
        # "category_class_weights": [0.2, 0.3, 0.6, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0],

        # Dual optimizer settings
        # "use_dual_optimizer": True,
        "encoder_lr": 1e-4,

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
        
        "encoder_filename":                 "encoder_model.keras",
                
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