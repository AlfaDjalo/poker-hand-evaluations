import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from ML.models import *
from ML.data.generators import AdaptedGenerator, build_output_adapter

def load_model(path):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            'CardSetEncoder': CardSetEncoder,
            'CardStateEncoder': CardStateEncoder,
            "GridValueHead": GridValueHead,
            'CardStateGridValueHead': CardStateGridValueHead,
            "EmbeddingValueHead": EmbeddingValueHead,
            'PairwiseComparisonHead': PairwiseComparisonHead,
            'CardStatePairwiseComparisonHead': CardStatePairwiseComparisonHead,
            "HandCategoryHead": HandCategoryHead,
            "CardStateHandCategoryHead": CardStateHandCategoryHead,
            "WeightedCategoricalCrossentropy": WeightedCategoricalCrossentropy,
        },
        compile=False,   # load without attempting to deserialize/compile saved compile config
        safe_mode=False
    )


def get_custom_objects():
    """Return all custom classes used in Poker models for loading Keras files."""
    return {
        'CardSetEncoder': CardSetEncoder,
        'CardStateEncoder': CardStateEncoder,
        "GridValueHead": GridValueHead,
        'CardStateGridValueHead': CardStateGridValueHead,
        "EmbeddingValueHead": EmbeddingValueHead,
        'PairwiseComparisonHead': PairwiseComparisonHead,
        'CardStatePairwiseComparisonHead': CardStatePairwiseComparisonHead,
        "HandCategoryHead": HandCategoryHead,
        "CardStateHandCategoryHead": CardStateHandCategoryHead,
        "WeightedCategoricalCrossentropy": WeightedCategoricalCrossentropy,
        # 'SuitPermutationLayer': SuitPermutationLayer,
    }

def train_embeddings(config=None, mode_config=None):
    """Handles model creation, data, compilation, and training."""
    mode = config["mode"]
    metrics = None
    
    # --- Data ---
    gen_cls = mode_config["generator"]
    gen_kwargs = mode_config.get("generator_kwargs", {})
    train_gen = gen_cls(config, **gen_kwargs)

    output_adapter = mode_config.get("output_adapter", {})
    adapter = build_output_adapter(output_adapter)
    train_gen = AdaptedGenerator(train_gen, adapter)

    # --- Paths ---
    save_dir = config["save_directory"]
    os.makedirs(save_dir, exist_ok=True)
    
    encoder_path = os.path.join(save_dir, config["encoder_filename"])
    model_path = os.path.join(save_dir, mode_config["save_file"])

    # --- Model creation or load ---
    if config["load_head_model"]:
        print(f"🔄 Loading model from {model_path}")
        model = load_model(model_path)
    else:
        print("🧱 Building new model...") 
        build_function_cls = mode_config["build_function"]
        # build_function_kwargs = training_mode_config.get("build_function_kwargs", {})
        model = build_function_cls(config)

    if config["load_encoder_model"]:
        # load_all_encoders_into_model(model, hand_encoder_path, board_encoder_path, combined_encoder_path, shared_encoder_path)
        load_encoder(model, encoder_path)

    # --- Compile ---
    loss_cfg = mode_config["loss_function"]
    loss = loss_cfg(config) if callable(loss_cfg) else loss_cfg
    loss_weights = mode_config["loss_weights"]
    lr = mode_config["learning_rate"]
    metrics = mode_config.get("metrics", None)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # include metrics if set
    if metrics is not None:
        model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)
    else:
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
            verbose=1
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
        dummy = tf.zeros((1, 14, 4, 1))
        enc.card_set_encoder(dummy)

        print("\n=== ENCODER ===")
        enc.card_set_encoder.summary()

    else:
        print("No encoder.")

    # --- Create validation generator ---
    val_config = dict(config)
    val_config["is_validation"] = True  # ✅ Flag as validation mode
    
    val_gen = gen_cls(val_config)
    val_gen = AdaptedGenerator(val_gen, adapter)

    # ✅ Pre-load ONE large validation batch from database
    print("📊 Pre-loading validation batch...")
    val_gen.preload_validation_data()
        
    # Print validation data size
    if isinstance(val_gen._preloaded_x, tuple):
        print(f"✅ Loaded {len(val_gen._preloaded_x[0])} validation samples (pairwise)")
    else:
        print(f"✅ Loaded {len(val_gen._preloaded_x)} validation samples ({mode})")
    print(f"✅ Validation y shape: {val_gen._preloaded_y.shape}")

    # --- Train ---
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config["epochs"],
        steps_per_epoch=config["steps_per_epoch"],
        callbacks=callbacks,
        verbose=1,
    )

    if config["save_model"]:
        print(f"💾 Saving model to {model_path}")
        model.save(model_path)
        save_encoder(model, encoder_path)
    
    return model

def get_encoder_from_model(model):
    """Return the shared CardStateEncoder if present."""
    for layer in model.layers:
        if isinstance(layer, CardStateEncoder):
           return layer
    for sublayer in getattr(layer, "layers", []):
        if isinstance(sublayer, CardStateEncoder):
           return sublayer
    return None

def find_encoder(model):
    if isinstance(model, CardStateEncoder):
        return model
    if hasattr(model, "layers"):
        for layer in model.layers:
            found = find_encoder(layer)
            if found is not None:
                return found
    return None


def save_encoder(model, path):
    """Save encoder weights to path if encoder exists."""
    encoder = get_encoder_from_model(model)
    model.summary()
    if encoder:
        print(f"💾 Saving shared encoder weights to {path}")
        dummy = tf.zeros((1, 14, 4, 1))
        # dummy = tf.zeros((1, 13, 4, 2))
        encoder(dummy, training=False)
        print(model.built)
        encoder.save(path)
    else:
        print("⚠️ No encoder found to save.")


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
        dummy_input = tf.zeros((1, 14, 4, 1))  # Direct shape, not unpacking
        try:
            _ = loaded_encoder(dummy_input, training=False)
            print("✅ Loaded encoder built successfully")
        except Exception as e:
            print(f"⚠️ Failed to build loaded encoder: {e}")
            return
        
    encoder = get_encoder_from_model(model)  
    encoder.set_weights(loaded_encoder.get_weights())

    encoder.trainable = True
    for layer in encoder.layers:
        layer.trainable = True

    print("✅ Shared encoder weights loaded successfully.")


