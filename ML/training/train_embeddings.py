import sys, os
import tensorflow as tf
import numpy as np
from keras import ops
from keras.layers import Lambda

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from db_plo import DB_PLO, open_db
from data.db_loader import data_generator
from models.implementation import PokerCNNEncoder, PokerComboModel, PokerValueModel, SuitEquivariantLayer, PokerValueHeads

def main():
    db = open_db()
    # data = data_generator(db, db_batch_size=32000, model_batch_size=64)
    pre_data = data_generator(db, db_batch_size=32000, model_batch_size=64)
    data = wrapped_generator(pre_data)
    
    inputs = tf.keras.Input(shape=(13, 4, 2))

    # --- Split and combine grids ---
    hand = inputs[..., 0:1]         # (batch, 13, 4, 1)
    board = inputs[..., 1:2]        # (batch, 13, 4, 1)
    combo = Lambda(lambda x:ops.concatenate([x[0], x[1], x[0]+x[1]], axis=-1))([hand, board])
    # combo_input = tf.concat(
    #     [hand, board, hand + board], axis=-1
    # )                                # (batch, 13, 4, 3)

    encoder = PokerComboModel(embedding_dim=32)
    value_heads = PokerValueHeads(encoder, activation="sigmoid")

    hand_v, board_v, combined_v = value_heads([hand, board, combo], training=True, return_all=True)

    # value_model = PokerValueModel(encoder.combined_encoder, activation="sigmoid")
    # value_output = value_model(combo, training=True)

    model = tf.keras.Model(inputs=inputs, outputs=[hand_v, board_v, combined_v])
    model.compile(optimizer="adam", loss=["mse", "mse", "mse"], loss_weights=[0.3, 0.3, 0.4], metrics=["mae", "mae", "mae"])  # include metric

    # Training
    model.fit(
        data,
        steps_per_epoch=500,
        epochs=100,
        verbose=1
    )

    # --- Evaluation Section ---
    # 1️⃣ Get a fresh random batch for evaluation
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

    # mse = tf.keras.losses.MeanSquaredError()(y_eval, y_pred).numpy().item()
    # mae = tf.keras.losses.MeanAbsoluteError()(y_eval, y_pred).numpy().item()
    # corr = np.corrcoef(y_eval[0].numpy().squeeze(), y_pred[0].squeeze())[0, 1]

    # # 3️⃣ Compute and print summary stats
    # # mse = tf.keras.losses.MeanSquaredError()(y_eval, y_pred).numpy().item()
    # # mae = tf.keras.losses.MeanAbsoluteError()(y_eval, y_pred).numpy().item()
    # # corr = np.corrcoef(y_eval.squeeze(), y_pred.squeeze())[0, 1]

    # print("\n📊 Evaluation Results:")
    # print(f"  MSE: {mse:.6f}")
    # print(f"  MAE: {mae:.6f}")
    # print(f"  Correlation: {corr:.4f}")

    # --- Optional: inspect embeddings ---
    hand = x_eval[..., 0:1]
    board = x_eval[..., 1:1+1]
    combo = Lambda(lambda x:ops.concatenate([x[0], x[1], x[0]+x[1]], axis=-1))([hand, board])
    # combo_input = np.concatenate([hand, board, hand + board], axis=-1)
    
    # hand_emb, board_emb, combined_emb = encoder(inputs, training=True, return_all=True)
    
    combined_encoder = encoder.combined_encoder
    combined_embeddings = combined_encoder.predict(combo, batch_size=64)
    print("\nCombined Embedding sample shape:", combined_embeddings.shape)
    print("First embedding vector:", np.array2string(combined_embeddings[0], precision=2))

    hand_encoder = encoder.hand_encoder
    hand_embeddings = hand_encoder.predict(hand, batch_size=64)
    print("\nHand Embedding sample shape:", hand_embeddings.shape)
    print("First embedding vector:", np.array2string(hand_embeddings[0], precision=2))

    board_encoder = encoder.board_encoder
    board_embeddings = board_encoder.predict(board, batch_size=64)
    print("\nBoard Embedding sample shape:", board_embeddings.shape)
    print("First embedding vector:", np.array2string(board_embeddings[0], precision=2))

    num_samples = min(25, y_eval[0].shape[0])
    idx = np.random.choice(num_samples, num_samples, replace=False)
    # idx = np.random.choice(len(y_eval), num_samples, replace=False)

    print("\nSample predictions vs actuals:")
    for i in idx:
        # print(f"  {i:3d}: predicted={int(1 + 7461*y_pred[i][0])}, actual={int(1+7461*y_eval[i][0])}")
        print(f"  {i:3d}: predicted={int(1 + 7461 * y_pred[2][i][0])}, actual={int(1 + 7461 * y_eval[2][i][0])}")



    # model.save("models/poker_value_model.keras")
    # encoder.save("models/poker_encoder.keras")

    # np.save("embeddings_sample.npy", embeddings)



def wrapped_generator(gen):
    for x, y in gen:
        yield x, (y, y, y)
# def main():

#     db = open_db()
#     data = data_generator(db, db_batch_size=10, model_batch_size=3)

#     encoder = PokerCNNEncoder(input_shape=(13, 4, 1))
#     model = PokerValueModel(encoder, activation="sigmoid")

#     model.compile(optimizer="adam", loss="mse")

#     model.fit(data, steps_per_epoch=100, epochs=5)

#     x_batch, _ = next(data)
#     embeddings = encoder.predict(x_batch)

#     print("Embeddings shape:", embeddings.shape)
#     print(embeddings[:2])



if __name__ == "__main__":
    main()
