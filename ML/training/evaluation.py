import tensorflow as tf
import numpy as np

def evaluate_model(model, data):
    x_eval, y_eval = next(data)

    # 2️⃣ Run predictions
    y_pred = model.predict(x_eval)

    y_eval = tuple(y_eval)
    y_pred = tuple(y_pred)

    # Overall weighted loss metrics (all three outputs)
    mse = tf.keras.losses.MeanSquaredError()(y_eval, y_pred).numpy().item()
    mae = tf.keras.losses.MeanAbsoluteError()(y_eval, y_pred).numpy().item()
    # corr = np.corrcoef(y_eval[0].numpy().squeeze(), y_pred[0].squeeze())[0, 1]
    corr = np.corrcoef(
        np.array(y_eval[0]).squeeze(),
        np.array(y_pred[0]).squeeze()
    )[0, 1]

    # Metrics for combo output only (index 2)
    mse_combo = tf.keras.losses.MeanSquaredError()(y_eval[2], y_pred[2]).numpy().item()
    mae_combo = tf.keras.losses.MeanAbsoluteError()(y_eval[2], y_pred[2]).numpy().item()
    # corr_combo = np.corrcoef(y_eval[2].numpy().squeeze(), y_pred[2].squeeze())[0, 1]
    corr_combo = np.corrcoef(
        np.array(y_eval[2]).squeeze(),
        np.array(y_pred[2]).squeeze()
    )[0, 1]

    print("\n📊 Evaluation Results:")
    print(f"  MSE (all): {mse:.6f}")
    print(f"  MAE (all): {mae:.6f}")
    print(f"  Correlation (all): {corr:.4f}")

    print("\n📊 Combo Output Metrics:")
    print(f"  MSE (combo): {mse_combo:.6f}")
    print(f"  MAE (combo): {mae_combo:.6f}")
    print(f"  Correlation (combo): {corr_combo:.4f}")

    return 
