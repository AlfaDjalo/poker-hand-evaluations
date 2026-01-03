import tensorflow as tf
import numpy as np
from tqdm import tqdm  # for progress bar
import itertools

def _collect_batches(data_iter, num_examples):
    """
    Pull batches from an iterator until we've accumulated >= num_examples
    (or the iterator is exhausted). Returns concatenated (x, y) lists.
    Expects data_iter to be an iterator that yields (x, y).
    """
    x_acc = []
    y_acc = []
    collected = 0
    while collected < num_examples:
        try:
            x_batch, y_batch = next(data_iter)
        except StopIteration:
            break
        # x_batch may be tuple (pairwise) or single array; keep as-is
        x_acc.append(x_batch)
        y_acc.append(y_batch)
        # infer batch size (handle pairwise tuple case)
        if isinstance(y_batch, tuple):
            bsz = y_batch[0].shape[0]
        else:
            bsz = y_batch.shape[0]
        collected += bsz
    if not x_acc:
        raise ValueError("No validation data available from iterator")
    # For simplicity return lists of batches (caller responsible for concatenation)
    return x_acc, y_acc


def evaluate_pairwise_comparison(model, data, num_examples=10):
    """
    Evaluate pairwise comparison model performance and print example predictions.

    Args:
        model: tf.keras.Model to evaluate.
        data: Iterator or generator yielding (inputs, labels).
        num_examples: Number of examples to show.

    Returns:
        None
    """
    # Collect enough batches to reach num_examples
    x_batches, y_batches = _collect_batches(data, num_examples)

    preds_list = []
    trues_list = []

    for x_batch, y_batch in zip(x_batches, y_batches):
        y_pred_batch = model.predict(x_batch)

        # Ensure output is numpy array, not a list/tuple
        if isinstance(y_pred_batch, (list, tuple)):
            # If multi-output model, assume only first output matters for pairwise
            y_pred_batch = y_pred_batch[0]

        preds_list.append(np.asarray(y_pred_batch))
        trues_list.append(np.asarray(y_batch))

    # Concatenate all predictions and true labels
    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(trues_list, axis=0)

    # Flatten predictions and labels to 1D if necessary
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    # Compute binary crossentropy loss
    bce_fn = tf.keras.losses.BinaryCrossentropy()
    bce_loss = bce_fn(y_true, y_pred).numpy()

    # Compute accuracy (threshold at 0.5)
    accuracy = np.mean((y_pred > 0.5) == (y_true > 0.5))

    print("\n==============================")
    print("🔍 Pairwise Comparison Evaluation")
    print("==============================")
    print(f"BCE Loss: {bce_loss:.6f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Print example predictions vs actuals
    print("\n🔎 Example predictions:")
    n = min(num_examples, len(y_pred))
    for i in range(n):
        print(f"{i+1:02d}: Pred={y_pred[i]:.3f}  |  Actual={int(y_true[i])}")

    # Build result dict
    # examples = []
    # n = min(num_examples, len(y_pred[0]))
    # for i in range(n):
    #     examples.append({
    #         # "p0": float(y_pred[0][i]), "a0": int(y_eval[0][i]),
    #         # "p1": float(y_pred[1][i]), "a1": int(y_eval[1][i]),
    #         # "p2": float(y_pred[2][i]), "a2": int(y_eval[2][i]),
    #     })

    return {
        "bce_loss": float(bce_loss),
        "accuracy": float(accuracy),
        "n_examples": n,
        # "examples": examples,
    }

def evaluate_hand_category(model, data, num_examples=10, full_confusion=False):
    """
    Evaluate hand category model performance and print predictions vs actual.

    Args:
        model: trained model
        data: data generator or iterator yielding (inputs, labels)
        num_examples: number of prediction examples to print
        full_confusion: if True, compute and print full confusion matrix

    Returns:
        None
    """
    # Collect enough examples across batches
    x_batches, y_batches = _collect_batches(data, num_examples)

    preds_list = []
    trues_list = []
    for x_batch, y_batch in zip(x_batches, y_batches):
        y_pred_batch = model.predict(x_batch)

        # If output is a list/tuple with single element, unwrap it
        if isinstance(y_pred_batch, (list, tuple)) and len(y_pred_batch) == 1:
            y_pred_batch = y_pred_batch[0]

        preds_list.append(np.asarray(y_pred_batch))
        trues_list.append(np.asarray(y_batch))

    # Concatenate all predictions and labels
    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(trues_list, axis=0)

    cce = tf.keras.losses.CategoricalCrossentropy()
    total_loss = cce(y_true, y_pred).numpy().item()

    def argmax_arr(x):
        arr = np.array(x)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        return np.argmax(arr, axis=-1)

    accuracy = np.mean(argmax_arr(y_pred) == argmax_arr(y_true))

    print("\n📊 Hand Category Evaluation Results:")
    print(f"  Categorical CE Loss (avg heads): {total_loss:.6f}")
    print(f"  Accuracy (all heads): {accuracy:.4f}")

    # ---- Confusion matrix ----
    if full_confusion:
        cm = evaluate_full_confusion(model)
        print("Full confusion matrix computed:")
        print(cm)
        # return
    else:
        y_true_cls = argmax_arr(y_true)
        y_pred_cls = argmax_arr(y_pred)

        n_classes = max(y_true_cls.max(), y_pred_cls.max()) + 1

        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true_cls, y_pred_cls):
            cm[int(t), int(p)] += 1

        row_sums = cm.sum(axis=1, keepdims=True)

        cm_norm = np.zeros_like(cm, dtype=float)
        np.divide(
            cm,
            row_sums,
            out=cm_norm,
            where=row_sums != 0
        )

        print("\n📈 Confusion Matrix (rows=actual, cols=predicted):")
        header = "     " + " ".join(f"{i:>5}" for i in range(n_classes))
        print(header)

        for i in range(n_classes):
            counts = " ".join(f"{cm[i, j]:5d}" for j in range(n_classes))
            pct = " ".join(f"{cm_norm[i, j] * 100:5.1f}%" for j in range(n_classes))
            print(f"{i:2d}: {counts}   | {pct}")

    # small examples (predicted class, true class)
    n = min(num_examples, y_true.shape[0])
    examples = []
    for i in range(n):
        examples.append({
            "pred_class": int(argmax_arr(y_pred)[i]),
            "true_class": int(argmax_arr(y_true)[i])
        })

    # optionally include confusion matrix
    return {
        "total_loss": float(total_loss),
        "accuracy": float(accuracy),
        # "total_loss": float(total_loss),
        "n_examples": n,
        "examples": examples,
        "confusion_matrix": cm.tolist() if not full_confusion else None,
    }


def evaluate_value(model, data, num_examples=10):
    """
    Evaluate embedding_value model which outputs a single tensor.

    Args:
        model: trained model
        data: data generator
        num_examples: how many prediction examples to print
    """
    x_batches, y_batches = _collect_batches(data, num_examples)

    preds_list = []
    trues_list = []
    for x_batch, y_batch in zip(x_batches, y_batches):
        y_pred_batch = model.predict(x_batch)

        # If output is a list/tuple with single element, unwrap it
        if isinstance(y_pred_batch, (list, tuple)) and len(y_pred_batch) == 1:
            y_pred_batch = y_pred_batch[0]

        preds_list.append(np.asarray(y_pred_batch))
        trues_list.append(np.asarray(y_batch))

    # Concatenate all predictions and labels
    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(trues_list, axis=0)

    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    # y_eval and y_pred are single tensors, not tuples
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred).numpy().item()
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred).numpy().item()

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    # corr = np.corrcoef(
    #     np.array(y_true).squeeze(),
    #     np.array(y_pred).squeeze()
    # )[0, 1]

    print("\n📊 Embedding Value Evaluation Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Corr: {corr:.4f}")

    # Undo normalization function
    # def denorm(x):
    #     return x * 7461.0 + 1.0

    print("\n🔎 Example predictions (value head):")
    n = min(num_examples, len(y_pred))
    # y_pred_main = y_pred.squeeze()
    # y_true_main = y_true.squeeze()

    for i in range(n):
        pred_val = denorm(float(y_pred[i]))
        actual_val = denorm(float(y_true[i]))
        print(f"{i+1:02d}: Pred={pred_val:.1f}  |  Actual={actual_val:.1f}")

    examples = []
    n = min(num_examples, len(y_pred))
    for i in range(n):
        examples.append({
            "pred": float(y_pred[i]),
            "actual": float(y_true[i])
        })

    return {
        "mse": float(mse),
        "mae": float(mae),
        "corr": float(corr),
        "n_examples": n,
        "examples": examples,
    }


def evaluate_full_confusion(model, batch_size=4096):
    """
    Evaluate the model over all 5-card hands, build confusion matrix.
    Assumes model returns a single output tensor with class probabilities.

    Args:
        model: keras model with single output (batch_size, num_classes)
        batch_size: batch size for prediction
        
    Returns:
        confusion_matrix: np.array shape (9,9)
    """
    num_classes = 9
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    all_hands = generate_all_5card_hands()

    inputs = []
    true_classes = []

    for hand in tqdm(all_hands, total=2598960):
        grid = hand_to_grid(hand)  # (14,4,2)
        inputs.append(grid)

        true_class = hand_to_true_class(hand)
        true_classes.append(true_class)

        if len(inputs) == batch_size:
            batch_x = np.stack(inputs)
            preds = model.predict(batch_x, verbose=0)  # Single output tensor
            
            pred_classes = np.argmax(preds, axis=1)
            
            for true_c, pred_c in zip(true_classes, pred_classes):
                confusion[true_c, pred_c] += 1

            inputs = []
            true_classes = []

    # Last batch if any
    if inputs:
        batch_x = np.stack(inputs)
        preds = model.predict(batch_x, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        for true_c, pred_c in zip(true_classes, pred_classes):
            confusion[true_c, pred_c] += 1

    return confusion


def generate_all_5card_hands(deck=range(52)):
    # deck is cards 0..51 representing a standard deck
    # returns generator of tuples (card1, card2, card3, card4, card5)
    return itertools.combinations(deck, 5)

def hand_to_grid(hand):
    """
    Converts a 5-card hand (tuple of ints 0-51) into a (14,4,2) input tensor
    matching the training format with duplicated Ace row.
    
    Channel 0: hand cards marked as 1
    Channel 1: board cards marked as 0 (empty)
    
    Card encoding: 
    - ranks 2..A mapped to 0..12 on axis 0
    - Ace (rank 0) is duplicated as row 0 AND row 13 for wheel straights
    - suits 0..3 mapped to axis 1
    Card index formula: card = rank + suit * 13
    """
    # Start with standard 13 ranks
    hand_grid = np.zeros((13, 4), dtype=np.float32)
    board_grid = np.zeros((13, 4), dtype=np.float32)
    
    for card in hand:
        rank = card % 13
        suit = card // 13
        hand_grid[rank, suit] = 1.0
    
    # Duplicate Ace row (index 0) to create 14x4 grids
    # This handles both Ace-high and Ace-low (wheel) straights
    ace_row_hand = hand_grid[0:1]  # Shape (1, 4)
    hand_grid = np.concatenate([ace_row_hand, hand_grid], axis=0)  # Now (14, 4)
    
    ace_row_board = board_grid[0:1]
    board_grid = np.concatenate([ace_row_board, board_grid], axis=0)  # Now (14, 4)
    
    # Stack as channels: (14, 4, 2)
    combined = np.stack([hand_grid, board_grid], axis=-1)
    
    return combined


def hand_to_true_class(hand):
    """
    Convert a 5-card hand (tuple of ints 0..51) to a category class 0..8.
    Category mapping follows CategoryGenerator.rank_to_category:
      0: Straight Flush
      1: Quads
      2: Full House
      3: Flush
      4: Straight
      5: Trips
      6: Two Pair
      7: One Pair
      8: No Pair (High Card)
    """
    # Extract rank (0=A ... 12=2) and suit (0..3)
    ranks = [c % 13 for c in hand]
    suits = [c // 13 for c in hand]

    # Map ranks to values where Ace is high (14) for straight detection
    rank_to_value = {0: 14, 1: 13, 2: 12, 3: 11, 4: 10, 5: 9, 6: 8, 7: 7, 8: 6, 9: 5, 10: 4, 11: 3, 12: 2}
    values = [rank_to_value[r] for r in ranks]

    # Multiplicity counts
    from collections import Counter
    cnt = Counter(ranks)
    counts = sorted(cnt.values(), reverse=True)  # e.g. [3,2] for full house

    # Flush if all suits identical
    is_flush = len(set(suits)) == 1

    # Straight detection (handle wheel A-2-3-4-5)
    unique_vals = sorted(set(values))
    is_straight = False
    if len(unique_vals) == 5:
        if unique_vals[-1] - unique_vals[0] == 4:
            is_straight = True
        # wheel: A(14),5,4,3,2
        if set(unique_vals) == {14, 5, 4, 3, 2}:
            is_straight = True

    # Decide category following priority
    if is_straight and is_flush:
        return 0  # Straight Flush
    if counts[0] == 4:
        return 1  # Quads
    if counts[0] == 3 and counts[1] == 2:
        return 2  # Full House
    if is_flush:
        return 3  # Flush
    if is_straight:
        return 4  # Straight
    if counts[0] == 3:
        return 5  # Trips
    if counts[0] == 2 and counts[1] == 2:
        return 6  # Two Pair
    if counts[0] == 2:
        return 7  # One Pair
    return 8      # No Pair (High Card)

def denorm(x):
    return x * 7461.0 + 1.0
