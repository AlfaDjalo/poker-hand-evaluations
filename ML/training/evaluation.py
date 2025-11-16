import tensorflow as tf
import numpy as np

def evaluate_model(model, data, model_type='absolute'):
    """
    Evaluate model performance.
    
    Args:
        model: trained model
        data: data generator
        model_type: 'absolute' or 'pairwise'
    """
    x_eval, y_eval = next(data)
    y_pred = model.predict(x_eval)
    
    y_eval = tuple(y_eval)
    y_pred = tuple(y_pred)
    
    if model_type == 'pairwise':
        # Binary classification metrics
        bce = tf.keras.losses.BinaryCrossentropy()(y_eval, y_pred).numpy().item()
        
        # Accuracy for all outputs
        acc = np.mean([
            np.mean((y_pred[i] > 0.5) == (y_eval[i] > 0.5))
            for i in range(3)
        ])
        
        # Combo-specific metrics
        bce_combo = tf.keras.losses.BinaryCrossentropy()(y_eval[2], y_pred[2]).numpy().item()
        acc_combo = np.mean((y_pred[2] > 0.5) == (y_eval[2] > 0.5))
        
        print("\n📊 Pairwise Evaluation Results:")
        print(f"  BCE Loss (all): {bce:.6f}")
        print(f"  Accuracy (all): {acc:.4f}")
        print(f"\n  BCE Loss (combo): {bce_combo:.6f}")
        print(f"  Accuracy (combo): {acc_combo:.4f}")
        
    else:  # absolute value model
        # Your existing regression metrics
        mse = tf.keras.losses.MeanSquaredError()(y_eval, y_pred).numpy().item()
        mae = tf.keras.losses.MeanAbsoluteError()(y_eval, y_pred).numpy().item()
        corr = np.corrcoef(np.array(y_eval[0]).squeeze(), np.array(y_pred[0]).squeeze())[0, 1]
        
        mse_combo = tf.keras.losses.MeanSquaredError()(y_eval[2], y_pred[2]).numpy().item()
        mae_combo = tf.keras.losses.MeanAbsoluteError()(y_eval[2], y_pred[2]).numpy().item()
        corr_combo = np.corrcoef(np.array(y_eval[2]).squeeze(), np.array(y_pred[2]).squeeze())[0, 1]
        
        print("\n📊 Absolute Value Evaluation Results:")
        print(f"  MSE (all): {mse:.6f}")
        print(f"  MAE (all): {mae:.6f}")
        print(f"  Correlation (all): {corr:.4f}")
        print(f"\n  MSE (combo): {mse_combo:.6f}")
        print(f"  MAE (combo): {mae_combo:.6f}")
        print(f"  Correlation (combo): {corr_combo:.4f}")

# def evaluate_model(model, data):
#     x_eval, y_eval = next(data)

#     # 2️⃣ Run predictions
#     y_pred = model.predict(x_eval)

#     y_eval = tuple(y_eval)
#     y_pred = tuple(y_pred)

#     # Overall weighted loss metrics (all three outputs)
#     mse = tf.keras.losses.MeanSquaredError()(y_eval, y_pred).numpy().item()
#     mae = tf.keras.losses.MeanAbsoluteError()(y_eval, y_pred).numpy().item()
#     # corr = np.corrcoef(y_eval[0].numpy().squeeze(), y_pred[0].squeeze())[0, 1]
#     corr = np.corrcoef(
#         np.array(y_eval[0]).squeeze(),
#         np.array(y_pred[0]).squeeze()
#     )[0, 1]

#     # Metrics for combo output only (index 2)
#     mse_combo = tf.keras.losses.MeanSquaredError()(y_eval[2], y_pred[2]).numpy().item()
#     mae_combo = tf.keras.losses.MeanAbsoluteError()(y_eval[2], y_pred[2]).numpy().item()
#     # corr_combo = np.corrcoef(y_eval[2].numpy().squeeze(), y_pred[2].squeeze())[0, 1]
#     corr_combo = np.corrcoef(
#         np.array(y_eval[2]).squeeze(),
#         np.array(y_pred[2]).squeeze()
#     )[0, 1]

#     print("\n📊 Evaluation Results:")
#     print(f"  MSE (all): {mse:.6f}")
#     print(f"  MAE (all): {mae:.6f}")
#     print(f"  Correlation (all): {corr:.4f}")

#     print("\n📊 Combo Output Metrics:")
#     print(f"  MSE (combo): {mse_combo:.6f}")
#     print(f"  MAE (combo): {mae_combo:.6f}")
#     print(f"  Correlation (combo): {corr_combo:.4f}")

#     return 
