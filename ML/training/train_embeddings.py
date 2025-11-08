import sys, os
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from db_plo import DB_PLO, open_db
from data.db_loader import data_generator
from models.implementation import PokerCNNEncoder, PokerValueModel, SuitEquivariantLayer

def main():
    db = open_db()
    data = data_generator(db, db_batch_size=32000, model_batch_size=64)

    inputs = tf.keras.Input(shape=(13, 4, 3))
    x = SuitEquivariantLayer(pooling="mean")(inputs)
    encoder = PokerCNNEncoder(input_shape=(13, 4, 3))
    value = PokerValueModel(encoder, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=value)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])  # include metric

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

    # 3️⃣ Compute and print summary stats
    mse = tf.keras.losses.MeanSquaredError()(y_eval, y_pred).numpy().item()
    mae = tf.keras.losses.MeanAbsoluteError()(y_eval, y_pred).numpy().item()
    corr = np.corrcoef(y_eval.squeeze(), y_pred.squeeze())[0, 1]

    print("\n📊 Evaluation Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Correlation: {corr:.4f}")

    # --- Optional: inspect embeddings ---
    embeddings = encoder.predict(x_eval)
    print("\nEmbedding sample shape:", embeddings.shape)
    print("First embedding vector:", embeddings[0][:8])  # show first 8 dims

    num_samples = 25
    idx = np.random.choice(len(y_eval), num_samples, replace=False)

    print("\nSample predictions vs actuals:")
    for i in idx:
        print(f"  {i:3d}: predicted={int(1 + 7461*y_pred[i][0])}, actual={int(1+7461*y_eval[i][0])}")


    # model.save("models/poker_value_model.keras")
    # encoder.save("models/poker_encoder.keras")

    # np.save("embeddings_sample.npy", embeddings)


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
