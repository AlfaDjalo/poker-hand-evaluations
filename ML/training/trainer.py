import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# from ML.models.utils import load_model
from ML.models.implementation import build_value_model, build_pairwise_model
from ML.models.implementation import (
    PokerComboModel,
    PokerValueModel,
    PokerValueHeads,
    PokerCNNEncoder,
    SuitEquivariantLayer, 
    # PairwiseModel
    PairwiseComparisonModel,
    PairwiseComparisonHeads
)
from ML.data.generators import AbsoluteGenerator, PairwiseGenerator, AlternatingGenerator, create_tensor_grids

def load_value_model(path):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            'PokerComboModel': PokerComboModel,
            "PokerValueModel": PokerValueModel,
            'PokerValueHeads': PokerValueHeads,
            'PokerCNNEncoder': PokerCNNEncoder,
            'SuitEquivariantLayer': SuitEquivariantLayer,
            # 'PairwiseModel': PairwiseModel
            'PairwiseComparisonModel': PairwiseComparisonModel,
            'PairwiseComparisonHeads': PairwiseComparisonHeads
            },
        safe_mode=False
    )

# def load_pairwise_model(path):
#     return tf.keras.models.load_model(
#         path,
#         custom_objects={
#             'PokerComboModel': PokerComboModel,
#             "PokerValueModel": PokerValueModel,
#             'PokerValueHeads': PokerValueHeads,
#             'PokerCNNEncoder': PokerCNNEncoder,
#             'SuitEquivariantLayer': SuitEquivariantLayer,
#             'PairwiseModel': PairwiseModel
#             },
#         safe_mode=False
#     )

def get_custom_objects():
    """Return all custom classes used in Poker models for loading Keras files."""
    return {
        "PokerCNNEncoder": PokerCNNEncoder,
        "PokerComboModel": PokerComboModel,
        "PokerValueHeads": PokerValueHeads,
        "PokerValueModel": PokerValueModel,
        # 'PairwiseModel': PairwiseModel
        'PairwiseComparisonModel': PairwiseComparisonModel,
        'PairwiseComparisonHeads': PairwiseComparisonHeads,
        "SuitEquivariantLayer": SuitEquivariantLayer,
    }

def train_embeddings(config=None):
    """Handles model creation, data, compilation, and training."""
    mode = config["mode"]
    # --- Data ---
    if mode == "absolute_value":
        train_gen = AbsoluteGenerator(config)
    elif mode in ["board", "hand", "mix"]:
        train_gen = PairwiseGenerator(config)
        # train_gen = PairwiseGenerator(config, mode_override=mode)
    elif mode == "alternating":
        train_gen = AlternatingGenerator(config)

    # --- Paths ---
    save_dir = config["save_directory"]
    hand_encoder_path = os.path.join(save_dir, config["hand_encoder_filename"])
    board_encoder_path = os.path.join(save_dir, config["board_encoder_filename"])
    combined_encoder_path = os.path.join(save_dir, config["combined_encoder_filename"])
    encoder_path = os.path.join(save_dir, config["encoder_filename"])
    abs_model_path = os.path.join(save_dir, config["absolute_model_filename"])
    pairwise_model_path = os.path.join(save_dir, config["pairwise_model_filename"])

    os.makedirs(save_dir, exist_ok=True)

    # --- Model creation or load ---
    if config["load_head_model"]:
    # if config["load_model"] and os.path.exists(config["save_path"]):
        if mode == "absolute_value":       
            print(f"🔄 Loading model from {abs_model_path}")
            model = load_value_model(abs_model_path)
        else:
            print(f"🔄 Loading model from {pairwise_model_path}")
            model = load_value_model(pairwise_model_path)
    else:
        print("🧱 Building new model...") 
        if mode=="absolute_value":
            model = build_value_model(config)
        else:
            model = build_pairwise_model(config) 

    if config["load_encoder_model"]:
        load_all_encoders_into_model(model, hand_encoder_path, board_encoder_path, combined_encoder_path)
        # load_encoder(model, encoder_path)

    # --- Compile ---
    if mode=="absolute_value":
        loss = ["mse", "mse", "mse"]
        loss_weights = [0.3, 0.3, 0.4]
        lr = config["lr_absolute_value"]
        # loss = tf.keras.losses.MeanSquaredError()
    else:
        loss = [
            tf.keras.losses.BinaryCrossentropy(from_logits=False),
            tf.keras.losses.BinaryCrossentropy(from_logits=False),
            tf.keras.losses.BinaryCrossentropy(from_logits=False)
            ]
        loss_weights = config.get("pairwise_loss_weights", [0.3, 0.3, 0.4])
        lr = config["lr_pairwise"]

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    # ✅ Add verbose=1 to ReduceLROnPlateau to see when it triggers
    reduce_cfg = config["reduce_lr"]
    early_cfg = config["early_stopping"]
    ckpt_cfg = config["checkpoint"]

    callbacks = [
        ReduceLROnPlateau(
            monitor=reduce_cfg["monitor"],
            factor=reduce_cfg["factor"],
            patience=reduce_cfg["patience"],
            min_lr=reduce_cfg["min_lr"],
            verbose=1  # ✅ Ensure verbose is set
        ),
        EarlyStopping(
            monitor=early_cfg["monitor"],
            patience=early_cfg["patience"],
            min_delta=reduce_cfg["min_delta"],
            restore_best_weights=early_cfg["restore_best_weights"],
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(config["save_directory"], "best_model.keras"),
            monitor="val_loss",
            save_best_only=ckpt_cfg["save_best_only"],
            verbose=1
        )
    ]

    print("\n=== TRAINABLE WEIGHTS DEBUG ===")
    total_params = 0
    trainable_params = 0

    for layer in model.layers:
        layer_trainable = sum(tf.size(w).numpy() for w in layer.trainable_weights)
        layer_total = sum(tf.size(w).numpy() for w in layer.weights)
        
        if layer.name and ("value" in layer.name.lower() or "dense" in layer.name.lower()):
            print(f"Layer: {layer.name:40} | Trainable: {layer_trainable:10} | Total: {layer_total:10} | layer.trainable: {layer.trainable}")
        
        trainable_params += layer_trainable
        total_params += layer_total

    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.1f}%")
    print("=== END DEBUG ===\n")

    print("\n=== FULL MODEL SUMMARY ===")
    model.summary()

    print("\n=== ENCODER SUMMARY ===")
    # Print summary for each encoder inside PokerComboModel
    enc = find_encoder(model)
    # enc = get_encoder_from_model(model)

    if enc is not None:
        # Build each encoder manually
        dummy = tf.zeros((1, 13, 4, 1))
        enc.hand_encoder(dummy)
        enc.board_encoder(dummy)
        enc.combined_encoder(dummy)

        print("\n=== HAND ENCODER ===")
        enc.hand_encoder.summary()

        print("\n=== BOARD ENCODER ===")
        enc.board_encoder.summary()

        print("\n=== COMBINED ENCODER ===")
        enc.combined_encoder.summary()

    else:
        print("No encoder.")
    # # ============================
    # # DEBUG BLOCK: BATCH + FORWARD
    # # ============================

    # print("\n=== DEBUG: Checking one batch ===")

    # batch = next(iter(train_gen))
    # inputs, y = batch

    # # Show shapes
    # print("Inputs:")
    # if isinstance(inputs, dict):
    #     # New format (dict with 6 entries)
    #     for k, v in inputs.items():
    #         print(f"  {k}: {v.shape}")    
    # elif isinstance(inputs, (tuple, list)) and len(inputs) == 2:
    #     # Old format: (x1, x2), where each is a tuple of 3 tensors
    #     x1, x2 = inputs
    #     print("  x1:")
    #     for t in x1:
    #         print("    ", t.shape)
    #     print("  x2:")
    #     for t in x2:
    #         print("    ", t.shape)

    # else:
    #     print("  Unknown input structure:", type(inputs))

    # print("Y shapes:",
    #     [t.shape for t in y] if isinstance(y, (tuple, list)) else y.shape)

    # # Try a forward call (critical!)
    # print("\n=== DEBUG: Forward pass through model ===")
    # try:
    #     out = model(inputs, training=False)
    #     if isinstance(out, (tuple, list)):
    #         print("Forward outputs:", [o.shape for o in out])
    #     else:
    #         print("Forward output:", out.shape)
    # except Exception as e:
    #     print("❌ Forward pass failed:", e)

    # # Check if weights change for one training step
    # print("\n=== DEBUG: Training step check ===")
    # before = [w.numpy().copy() for w in model.weights]

    # try:
    #     model.train_on_batch(inputs, y)

    #     # Check if weights change for one training step
    #     print("\n=== DEBUG: Training step check ===")
        
    #     # Save weights before
    #     encoder_weights_before = [w.numpy().copy() for w in enc.trainable_weights]
    #     value_head_weights_before = [w.numpy().copy() for w in model.layers[-1].trainable_weights]  # last layer
        
    #     loss = model.train_on_batch((x1, x2), y)
        
    #     # Save weights after
    #     encoder_weights_after = model.get_weights()[:len(enc.trainable_weights)]
    #     value_head_weights_after = model.get_weights()[-len(model.layers[-1].trainable_weights):]
        
    #     encoder_changed = any((encoder_weights_before[i] != encoder_weights_after[i]).any() for i in range(len(encoder_weights_before)))
    #     value_changed = any((value_head_weights_before[i] != value_head_weights_after[i]).any() for i in range(len(value_head_weights_before)))
        
    #     print(f"Encoder weights changed: {encoder_changed}")
    #     print(f"Value head weights changed: {value_changed}")
    #     print(f"Loss: {loss}")



    #     after = model.get_weights()

    #     changed = any((before[i] != after[i]).any() for i in range(len(before)))
    #     print("Any weight changed?:", changed)

    # except Exception as e:
    #     print("❌ train_on_batch failed:", e)

    # print("=== END DEBUG ===\n")

    # --- Create validation generator ---
    val_config = dict(config)
    val_config["is_validation"] = True  # ✅ Flag as validation mode
    
    if mode == "absolute_value":
        val_gen = AbsoluteGenerator(val_config)
    elif mode in ["board", "hand", "mix"]:
        val_gen = PairwiseGenerator(val_config)
    elif mode == "alternating":
        val_gen = AlternatingGenerator(val_config)
    
    val_gen.preload_validation_data()
    
    # ✅ Pre-load ONE large validation batch from database
    print("📊 Pre-loading validation batch...")
    
    if mode == "absolute_value":
        sample_evals = val_gen.db.get_sample_evaluations(val_gen.db_batch_size)
        val_x, val_y = create_tensor_grids(mode, sample_evals)
    elif mode == "alternating":
        # ✅ For alternating mode, use "mix" for validation (balanced representation)
        sample_evals = val_gen.db.get_comparison_pairs("mix", val_gen.db_batch_size)
        val_x, val_y = create_tensor_grids("mix", sample_evals)
    else:
        # ✅ For "board", "hand", "mix" modes, use the same mode
        sample_evals = val_gen.db.get_comparison_pairs(mode, val_gen.db_batch_size)
        val_x, val_y = create_tensor_grids(mode, sample_evals)
    
    # Store in generator for slicing each epoch
    # ✅ Handle both tensor and tuple types
    if isinstance(val_x, tuple):
        # Pairwise: (x_A, x_B)
        val_gen._preloaded_x = (
            val_x[0].numpy() if not isinstance(val_x[0], np.ndarray) else val_x[0],
            val_x[1].numpy() if not isinstance(val_x[1], np.ndarray) else val_x[1]
        )
    else:
        # Absolute value: single tensor
        val_gen._preloaded_x = val_x.numpy() if not isinstance(val_x, np.ndarray) else val_x
    
    val_gen._preloaded_y = val_y.numpy() if not isinstance(val_y, np.ndarray) else val_y
    
    # Print validation data size
    if isinstance(val_gen._preloaded_x, tuple):
        print(f"✅ Loaded {len(val_gen._preloaded_x[0])} validation samples (pairwise)")
    else:
        print(f"✅ Loaded {len(val_gen._preloaded_x)} validation samples (absolute)")

    # --- Train ---
    model.fit(
        train_gen,
        validation_data=val_gen,  # ✅ Uses pre-loaded data internally
        epochs=config["epochs"],
        steps_per_epoch=config["steps_per_epoch"],
        callbacks=callbacks,
        verbose=1,
    )

    if config["save_model"]:
        if mode == "absolute_value":
            print(f"💾 Saving absolute model to {abs_model_path}")
            model.save(abs_model_path)
        else:
            print(f"💾 Saving pairwise model to {pairwise_model_path}")
            model.save(pairwise_model_path)
        save_all_encoders(model, hand_encoder_path, board_encoder_path, combined_encoder_path)
        # save_encoder(model, encoder_path)
    
    return model

def get_encoder_from_model(model):
    """Return the shared PokerComboModel if present."""
    for layer in model.layers:
        if isinstance(layer, PokerComboModel):
           return layer
    for sublayer in getattr(layer, "layers", []):
        if isinstance(sublayer, PokerComboModel):
           return sublayer
    return None

def find_encoder(model):
    if isinstance(model, PokerComboModel):
        return model
    if hasattr(model, "layers"):
        for layer in model.layers:
            found = find_encoder(layer)
            if found is not None:
                return found
    return None

# def save_encoder_weights(model, path):
#     """Extracts encoder from model and saves its weights."""
#     encoder = model.get_layer("poker_combo_model")
#     encoder.save_weights(path)

# def load_encoder_weights(model, path):
#     """Loads encoder weights into model if available."""
#     encoder = model.get_layer("poker_combo_model")
#     if os.path.exists(path):
#         print(f"🔄 Loading encoder weights from {path}")
#         encoder.load_weights(path)
#     else:
#         print(f"⚠️ Encoder weights not found at {path}")


def save_encoder(model, path):
    """Save encoder weights to path if encoder exists."""
    encoder = get_encoder_from_model(model)
    model.summary()
    if encoder:
        print(f"💾 Saving shared encoder weights to {path}")
        dummy = tf.zeros((1, 13, 4, 2))
        encoder(dummy, training=False)
        print(model.built)
        encoder.save(path)
    else:
        print("⚠️ No encoder found to save.")

def save_all_encoders(model, hand_encoder_path, board_encoder_path, combined_encoder_path):
    encoder = find_encoder(model)  # Get PokerComboModel
    if not encoder:
        print("⚠️ No encoder found to save")
        return

    # Save each encoder separately
    encoder.hand_encoder.save(hand_encoder_path)
    encoder.board_encoder.save(board_encoder_path)
    encoder.combined_encoder.save(combined_encoder_path)

    print("Saved hand, board and combined encoders.")

def load_all_encoders_into_model(model, hand_encoder_path, board_encoder_path, combined_encoder_path):
    hand = tf.keras.models.load_model(hand_encoder_path, compile=False)
    board = tf.keras.models.load_model(board_encoder_path, compile=False)
    combined = tf.keras.models.load_model(combined_encoder_path, compile=False)

    encoder = find_encoder(model)
    if not encoder:
        print("⚠️ No encoder found in model")
        return       

    encoder.hand_encoder.set_weights(hand.get_weights())
    encoder.board_encoder.set_weights(board.get_weights())
    encoder.combined_encoder.set_weights(combined.get_weights())

    print("✅ Loaded hand, board, and combined encoder weights into model")
    
    # return encoder

def load_encoder(model, encoder_path):
    """Load encoder weights into model if encoder exists and file found."""

    if not os.path.exists(encoder_path):
        print(f"⚠️ Encoder file not found at {encoder_path}")
        return

    print(f"🔄 Loading shared encoder weights from {encoder_path}")
    loaded_encoder = tf.keras.models.load_model(
        encoder_path,
        custom_objects = get_custom_objects(),
        compile = False
    )

    # Build the loaded encoder if not already built
    if not loaded_encoder.built:
        dummy_input = tf.zeros((1, 13, 4, 1))  # Direct shape, not unpacking
        try:
            _ = loaded_encoder(dummy_input, training=False)
            print("✅ Loaded encoder built successfully")
        except Exception as e:
            print(f"⚠️ Failed to build loaded encoder: {e}")
            return
        
    # if not loaded_encoder.built:
    #     try:
    #         input_shape = model.input_shape_encoder
    #         # input_shape = model.encoder_input_shape
    #     except AttributeError:
    #         # fallback: use default config value
    #         input_shape = (13, 4, 1)

    #     real_shape = loaded_encoder.hand_encoder.input_shape_encoder[1:]   # remove batch dim
    #     dummy_input = tf.zeros((1, *real_shape))
    #     _ = loaded_encoder(dummy_input, training=False)

        # dummy_input = tf.zeros((1, *input_shape))
        # _ = loaded_encoder(dummy_input, training=False)


    # if encoder and os.path.exists(path):
    encoder = get_encoder_from_model(model)  
    encoder.set_weights(loaded_encoder.get_weights())

    encoder.trainable = True
    for layer in encoder.layers:
        layer.trainable = True
    # elif not os.path.exists(path):
    #     print(f"⚠️ Encoder file not found at {path}")
    # else:
    #     print("⚠️ No encoder found in model to load into.")
    print("✅ Shared encoder weights loaded successfully.")


