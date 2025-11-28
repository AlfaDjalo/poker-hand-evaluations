import tensorflow as tf
import numpy as np


def evaluate_model(model, data, model_type='absolute', num_examples=20):
    """
    Evaluate model performance and print predictions vs actual.

    Args:
        model: trained model
        data: data generator
        model_type: 'absolute' or 'pairwise'
        num_examples: how many prediction examples to print
    """
    # 1) Fetch evaluation batch
    x_eval, y_eval = next(data)
    y_pred = model.predict(x_eval)

    # Convert to tuple so indexing is consistent
    y_eval = tuple(y_eval)
    y_pred = tuple(y_pred)

    print("\n==============================")
    print("🔍 Showing example predictions")
    print("==============================")

    # -----------------------------
    # PAIRWISE MODEL (3 heads, sigmoid)
    # -----------------------------
    if model_type == 'pairwise':

        # Binary cross-entropy over all outputs
        bce = tf.keras.losses.BinaryCrossentropy()(y_eval, y_pred).numpy().item()

        # Accuracy for each head
        acc = np.mean([
            np.mean((y_pred[i] > 0.5) == (y_eval[i] > 0.5))
            for i in range(3)
        ])

        print("\n📊 Pairwise Evaluation Results:")
        print(f"  BCE Loss (all heads): {bce:.6f}")
        print(f"  Accuracy (all heads): {acc:.4f}")

        # Combo-only metrics
        bce_combo = tf.keras.losses.BinaryCrossentropy()(y_eval[2], y_pred[2]).numpy().item()
        acc_combo = np.mean((y_pred[2] > 0.5) == (y_eval[2] > 0.5))

        print(f"\n  BCE Loss (combo): {bce_combo:.6f}")
        print(f"  Accuracy (combo): {acc_combo:.4f}")

        # ---- Example predictions ----
        print("\n🔎 Example predictions (all 3 outputs):")
        n = min(num_examples, len(y_pred[0]))

        for i in range(n):
            p0 = float(y_pred[0][i])
            a0 = int(y_eval[0][i])

            p1 = float(y_pred[1][i])
            a1 = int(y_eval[1][i])

            p2 = float(y_pred[2][i])
            a2 = int(y_eval[2][i])

            print(f"{i+1:02d}: "
                  f"p0={p0:.3f}/{a0}   "
                  f"p1={p1:.3f}/{a1}   "
                  f"p2={p2:.3f}/{a2}")

        print("\n🔎 Example predictions (combo head):")
        for i in range(n):
            p = float(y_pred[2][i])
            a = int(y_eval[2][i])
            print(f"{i+1:02d}: Pred={p:.3f}  |  Actual={a}")

        return

    # -----------------------------
    # VALUE MODEL (regression)
    # -----------------------------
    else:
        # Regression metrics (all 3 heads)
        mse = tf.keras.losses.MeanSquaredError()(y_eval, y_pred).numpy().item()
        mae = tf.keras.losses.MeanAbsoluteError()(y_eval, y_pred).numpy().item()
        corr = np.corrcoef(
            np.array(y_eval[0]).squeeze(),
            np.array(y_pred[0]).squeeze()
        )[0, 1]

        # Combo-only metrics
        mse_combo = tf.keras.losses.MeanSquaredError()(y_eval[2], y_pred[2]).numpy().item()
        mae_combo = tf.keras.losses.MeanAbsoluteError()(y_eval[2], y_pred[2]).numpy().item()
        corr_combo = np.corrcoef(
            np.array(y_eval[2]).squeeze(),
            np.array(y_pred[2]).squeeze()
        )[0, 1]

        print("\n📊 Absolute Value Evaluation Results:")
        print(f"  MSE (all): {mse:.6f}")
        print(f"  MAE (all): {mae:.6f}")
        print(f"  Corr (all): {corr:.4f}")

        print(f"\n  MSE (combo): {mse_combo:.6f}")
        print(f"  MAE (combo): {mae_combo:.6f}")
        print(f"  Corr (combo): {corr_combo:.4f}")

        # ---- Undo your normalization ----
        # original_value = pred * 7461 + 1
        def denorm(x):
            return x * 7461.0 + 1.0

        print("\n🔎 Example predictions (value head):")
        n = min(num_examples, len(y_pred[0]))

        y_pred_main = y_pred[0].squeeze()
        y_eval_main = y_eval[0].squeeze()

        for i in range(n):
            pred_val = denorm(float(y_pred_main[i]))
            actual_val = denorm(float(y_eval_main[i]))
            print(f"{i+1:02d}: Pred={pred_val:.1f}  |  Actual={actual_val:.1f}")

        print("\n🔎 Example predictions (combo head):")
        y_pred_combo = y_pred[2].squeeze()
        y_eval_combo = y_eval[2].squeeze()

        for i in range(n):
            pred_val = denorm(float(y_pred_combo[i]))
            actual_val = denorm(float(y_eval_combo[i]))
            print(f"{i+1:02d}: Pred={pred_val:.1f}  |  Actual={actual_val:.1f}")

