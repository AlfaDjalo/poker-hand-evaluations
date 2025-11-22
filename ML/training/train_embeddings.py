import sys, os
import tensorflow as tf
import numpy as np
from keras import ops
from keras.layers import Lambda

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

# from db_plo import DB_PLO, open_db
# from data.db_loader import data_generator, data_generator_3_heads
from data.generators import AbsoluteGenerator, PairwiseGenerator
from models.implementation import PokerCNNEncoder, PokerComboModel, PokerValueModel, SuitEquivariantLayer, PokerValueHeads, ComboConcatLayer
# from models.utils import save_model, load_model
from training.evaluation import evaluate_model
from training.trainer import train_embeddings
from config import get_config, summarize_config

def wrapped_generator(gen):
    for x, y in gen:
        yield x, (y, y, y)

def build_model():
    inputs = tf.keras.Input(shape=(13, 4, 2))
    
    # --- Split and combine grids ---
    hand = inputs[..., 0:1]         # (batch, 13, 4, 1)
    board = inputs[..., 1:2]        # (batch, 13, 4, 1)
    combo = ComboConcatLayer()([hand, board])
    # combo = Lambda(lambda x:ops.concatenate([x[0], x[1], x[0]+x[1]], axis=-1), output_shape=(13, 4, 3))([hand, board])

    encoder = PokerComboModel(embedding_dim=32)
    value_heads = PokerValueHeads(encoder, activation="sigmoid")

    hand_v, board_v, combined_v = value_heads([hand, board, combo], training=True, return_all=True)

    model = tf.keras.Model(inputs=inputs, outputs=[hand_v, board_v, combined_v])
    model.compile(optimizer="adam", loss=["mse", "mse", "mse"], loss_weights=[0.3, 0.3, 0.4], metrics=["mae", "mae", "mae"])  # include metric

    return model


def train_model(model, data, epochs=100, steps_per_epoch=500):
    # Training
    model.fit(
        data,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1
    )    
    return model

def evaluate_model_old(model, data):
    x_eval, y_eval = next(data)

    # 2️⃣ Run predictions
    y_pred = model.predict(x_eval)

    y_eval = tuple(y_eval)
    y_pred = tuple(y_pred)

    # Overall weighted loss metrics (all three outputs)
    mse = tf.keras.losses.MeanSquaredError()(y_eval, y_pred).numpy().item()
    mae = tf.keras.losses.MeanAbsoluteError()(y_eval, y_pred).numpy().item()
    corr = np.corrcoef(y_eval[0].numpy().squeeze(), y_pred[0].squeeze())[0, 1]

    # Metrics for combo output only (index 2)
    mse_combo = tf.keras.losses.MeanSquaredError()(y_eval[2], y_pred[2]).numpy().item()
    mae_combo = tf.keras.losses.MeanAbsoluteError()(y_eval[2], y_pred[2]).numpy().item()
    corr_combo = np.corrcoef(y_eval[2].numpy().squeeze(), y_pred[2].squeeze())[0, 1]

    print("\n📊 Evaluation Results:")
    print(f"  MSE (all): {mse:.6f}")
    print(f"  MAE (all): {mae:.6f}")
    print(f"  Correlation (all): {corr:.4f}")

    print("\n📊 Combo Output Metrics:")
    print(f"  MSE (combo): {mse_combo:.6f}")
    print(f"  MAE (combo): {mae_combo:.6f}")
    print(f"  Correlation (combo): {corr_combo:.4f}")

    return 

def main():

    config = get_config()
    summarize_config(config)
    model = train_embeddings(config=config)
    
    # --- Evaluate after training ---
    print("\n🔍 Running post-training evaluation...")
    eval_gen = AbsoluteGenerator(config) if config["mode"] == "absolute_value" else PairwiseGenerator(config, mode_override="hand")

    if config["mode"] == "absolute_value":
        evaluate_model(model, iter(eval_gen), model_type='absolute', num_examples=10)
    else:
        evaluate_model(model, iter(eval_gen), model_type='pairwise', num_examples=10)

if __name__ == "__main__":
    main()
