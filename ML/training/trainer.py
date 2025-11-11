import os
import tensorflow as tf

# from ML.models.utils import load_model
from models.implementation import build_value_model
from data.generators import AbsoluteGenerator, PairwiseGenerator
from models.implementation import (
    PokerComboModel,
    PokerValueHeads,
    PokerCNNEncoder,
    SuitEquivariantLayer, 
    build_value_model
)

def load_value_model(path):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            'PokerComboModel': PokerComboModel,
            'PokerValueHeads': PokerValueHeads,
            'PokerCNNEncoder': PokerCNNEncoder,
            'SuitEquivariantLayer': SuitEquivariantLayer
            },
        safe_mode=False
    )

def train_embeddings(mode="absolute_value", config=None):
    """Handles model creation, data, compilation, and training."""

    # --- Data ---
    if mode=="absolute_value":
        train_gen = AbsoluteGenerator(config)
    else:
        train_gen = PairwiseGenerator(config)

    # --- Model creation or load ---
    if config["load_model"] and os.path.exists(config["save_path"]):
        print(f"🔄 Loading model from {config["save_path"]}")
        model = load_value_model(config["save_path"])
    else:
        print("🧱 Building new model...") 
        model = build_value_model(config)

    # --- Compile ---
    if mode=="absolute_value":
        loss = ["mse", "mse", "mse"]
        loss_weights = [0.3, 0.3, 0.4]
        # loss = tf.keras.losses.MeanSquaredError()
    else:
        loss = tf.keras.losses.BinaryCrossEntropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["lr"])
    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)
    # model.compile(optimizer=optimizer, loss=loss)

    # --- Train ---
    model.fit(
        train_gen,
        epochs=config["epochs"],
        steps_per_epoch=config["steps_per_epoch"],
        callbacks=config["callbacks"],
        verbose=1,
    )

    if config["save_model"]:
        os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
        print(f"💾 Saving model to {config["save_path"]}")
        model.save(config["save_path"])

    return model