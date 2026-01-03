import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import copy, time, json

from ML.models import *
from ML.training.dual_optimizer import DualOptimizerModel

# Add this class at the top of your trainer.py file, after imports

class SaveBaseModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Custom checkpoint that saves the base_model from DualOptimizerModel,
    not the wrapper itself.
    """
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        # Check if this is the best model so far
        if not self.save_best_only or current < self.best:
            if self.save_best_only:
                self.best = current
            
            # Extract base_model if wrapped
            model_to_save = self.model
            if isinstance(model_to_save, DualOptimizerModel):
                model_to_save = model_to_save.base_model
            
            if self.verbose > 0:
                print(f'\nEpoch {epoch + 1}: {self.monitor} improved to {current:.5f}, saving model to {self.filepath}')
            
            model_to_save.save(self.filepath)

def load_model(path):
    """Load model, handling DualOptimizerModel wrapper gracefully."""
    # if not os.path.exists(path):
    #     raise FileNotFoundError(f"Model file not found: {path}")

    # 1) Try load as a plain base model (no DualOptimizerModel present in custom objects)
    custom_base = {k: v for k, v in get_custom_objects().items() if k != "DualOptimizerModel"}

    try:
        m = tf.keras.models.load_model(path, custom_objects=custom_base, compile=False, safe_mode=False)
        # If this succeeded and returned a wrapper (unlikely), unwrap if possible
        # if isinstance(m, DualOptimizerModel) and getattr(m, "base_model", None) is not None:
        #     base = m.base_model
        #     try:
        #         base.save(path)  # migrate file to base_model-only
        #         print("🔧 Migrated saved DualOptimizerModel -> base_model (overwrote file)")
        #     except Exception as e:
        #         print("⚠️ Migration save failed:", e)
        #     return base
        return m
    except Exception as primary_exc:
        # 2) Fallback: try loading including DualOptimizerModel (for legacy files)
        try:
            custom = get_custom_objects()
            m = tf.keras.models.load_model(path, custom_objects=custom, compile=False, safe_mode=False)
            # If we got a DualOptimizerModel wrapper, try to unwrap to its base_model.
            if isinstance(m, DualOptimizerModel):
                base = getattr(m, "base_model", None)
                # If base was reconstructed but has no weights/layers, warn and fail with helpful message.
                if base is None or len(getattr(base, "weights", [])) == 0:
                    raise RuntimeError(
                        "Saved model appears to be a DualOptimizerModel wrapper whose nested base_model "
                        "couldn't be reconstructed; weights can't be restored. To fix:\n"
                        "  1) In an environment that can deserialize the original wrapper, run:\n"
                        "       m = tf.keras.models.load_model('<path>', custom_objects={'DualOptimizerModel': DualOptimizerModel}, compile=False)\n"
                        "       m.base_model.save('<path>')\n"
                        "     This will overwrite the file with the underlying base_model (recommended).\n"
                        "  2) Or run trainer.migrate_dual_to_base('<path>') if you have access to the same code that originally saved it.\n"
                        "After migrating the saved file to contain the base model only, retry loading."
                    )
                # Otherwise return the unwrapped base model
                return base
            return m
        except Exception as fallback_exc:
            # Combine errors for diagnostics
            raise RuntimeError(f"Failed to load model '{path}': primary: {primary_exc}; fallback: {fallback_exc}")

# def load_model_old(path):
#     """Load model, handling DualOptimizerModel wrapper gracefully."""
#     # if not os.path.exists(path):
#     #     raise FileNotFoundError(f"Model file not found: {path}")

# 	# 1) Try load as a plain base model (no DualOptimizerModel present in custom objects)
# 	custom_base = {k: v for k, v in get_custom_objects().items() if k != "DualOptimizerModel"}
	
#     try:
# 		m = tf.keras.models.load_model(path, custom_objects=custom_base, compile=False, safe_mode=False)
# 		# If this succeeded and returned a wrapper (unlikely), unwrap if possible
# 		if isinstance(m, DualOptimizerModel) and getattr(m, "base_model", None) is not None:
# 			base = m.base_model
# 			try:
# 				base.save(path)  # migrate file to base_model-only
# 				print("🔧 Migrated saved DualOptimizerModel -> base_model (overwrote file)")
# 			except Exception as e:
# 				print("⚠️ Migration save failed:", e)
# 			return base
# 		return m
# 	except Exception as primary_exc:
# 		# 2) Fallback: try loading including DualOptimizerModel (for legacy files)
# 		try:
# 			custom = get_custom_objects()
# 			m = tf.keras.models.load_model(path, custom_objects=custom, compile=False, safe_mode=False)
# 			if isinstance(m, DualOptimizerModel) and getattr(m, "base_model", None) is not None:
# 				base = m.base_model
# 				try:
# 					base.save(path)  # rewrite saved file as base_model only to avoid future errors
# 					print("🔧 Migrated saved DualOptimizerModel -> base_model (overwrote file)")
# 				except Exception as e:
# 					print("⚠️ Migration save failed:", e)
# 				return base
# 			return m
# 		except Exception as fallback_exc:
# 			# Combine errors for diagnostics
# 			raise RuntimeError(f"Failed to load model '{path}': primary: {primary_exc}; fallback: {fallback_exc}")

def get_custom_objects():
    """Return all custom classes used in Poker models for loading Keras files."""
    return {
        'CardSetEncoder': CardSetEncoder,
        'CardStateEncoder': CardStateEncoder,
        "CombinedInputValueHead": CombinedInputValueHead,
        "SeparateInputValueHead": SeparateInputValueHead,
        "CombinedPairwiseComparisonValueHead": CombinedInputPairwiseComparisonHead,
        "SeparateInputPairwiseComparisonHead": SeparateInputPairwiseComparisonHead,
        # "DualOptimizerModel": DualOptimizerModel,
        # "WeightedCategoricalCrossentropy": WeightedCategoricalCrossentropy,
        # 'SuitPermutationLayer': SuitPermutationLayer,
    }

def train_embeddings(config=None, mode_config=None, return_info=False):
    """Handles model creation, data, compilation, and training."""
    mode = config["mode"]
    start_time = time.time()
    metrics = None
    
    # --- Data ---
    gen_cls = mode_config["generator"]
    gen_kwargs = mode_config.get("generator_kwargs", {})
    train_gen = gen_cls(config, **gen_kwargs)

    # output_adapter = mode_config.get("output_adapter", {})
    # adapter = build_output_adapter(output_adapter)
    # train_gen = AdaptedGenerator(train_gen, adapter)

    # --- Paths ---
    save_dir = config["save_directory"]
    os.makedirs(save_dir, exist_ok=True)
    
    encoder_path = os.path.join(save_dir, config["encoder_filename"])
    head_filename = config.get("submode", "combined") + "_" + mode_config["save_file"]
    # model_path = os.path.join(save_dir, model_filename)
    head_path = os.path.join(save_dir, head_filename)

    # --- Model creation or load ---
    if config["load_head_model"]:
        print(f"🔄 Loading model from {head_path}")
        base_model = load_model(head_path)
    else:
        print("🧱 Building new model...") 
        base_model = build_model(config, mode_config)
        
    if config["load_encoder_model"]:
        # print(f"🔄 Loading encoder weights from {encoder_path}")
        load_encoder(base_model, encoder_path)
        # load_encoder(model.base_model if isinstance(model, DualOptimizerModel) else model, encoder_path)

    encoder_lr = float(config.get("encoder_lr", 1e-4))
    head_lr = float(config.get("head_lr", 1e-3))
    model = DualOptimizerModel(base_model, encoder_lr=encoder_lr, head_lr=head_lr)


    # ✅ WRAP WITH DUAL OPTIMIZER (only if enabled in config)
    # 

    # use_dual_optimizer = config.get("use_dual_optimizer", False)
    # if use_dual_optimizer:
    #     encoder_lr = float(config.get("encoder_lr", config.get("encoder_lr", 1e-4)))
    #     head_lr = float(config.get("head_lr", mode_config.get("learning_rate", 1e-3)))
    # else:
    #     # Single optimizer uses head_lr if provided, otherwise mode_config LR
    #     head_lr = float(config.get("head_lr", mode_config.get("learning_rate", 1e-3)))
    #     encoder_lr = None

    # if use_dual_optimizer:
    #     print(f"\n🔧 Using dual optimizer:")
    #     print(f"  Encoder LR: {encoder_lr}")
    #     print(f"  Head LR: {head_lr}")
        
        # model = DualOptimizerModel(base_model, encoder_lr=encoder_lr, head_lr=head_lr)
    # else:
    #     model = base_model
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=head_lr)

    # --- Compile ---
    loss_cfg = mode_config["loss_function"]
    loss = loss_cfg(config) if callable(loss_cfg) else loss_cfg
    loss_weights = mode_config["loss_weights"]
    # lr = mode_config["learning_rate"]
    metrics = mode_config.get("metrics", None)

    if metrics is not None:
        model.compile(optimizer=None, loss=loss, loss_weights=loss_weights, metrics=metrics)
    else:
        model.compile(optimizer=None, loss=loss, loss_weights=loss_weights)
    # if use_dual_optimizer:
    #     # DualOptimizerModel handles optimizers internally
    #     if metrics is not None:
    #         model.compile(loss=loss, loss_weights=loss_weights, metrics=metrics)
    #     else:
    #         model.compile(loss=loss, loss_weights=loss_weights)
    # else:
    #     # Single optimizer (original behavior)
    #     if metrics is not None:
    #         model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)
    #     else:
    #         model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)
            
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # # include metrics if set
    # if metrics is not None:
    #     model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)
    # else:
    #     model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    # ✅ Add verbose=1 to ReduceLROnPlateau to see when it triggers
    reduce_cfg = config["reduce_lr"]
    early_cfg = config["early_stopping"]
    ckpt_cfg = config["checkpoint"]

    callbacks = [
        EarlyStopping(
            monitor=early_cfg["monitor"],
            patience=early_cfg["patience"],
            min_delta=reduce_cfg["min_delta"],
            restore_best_weights=early_cfg["restore_best_weights"],
            verbose=1
        ),
        # ✅ Use custom checkpoint that saves base_model
        SaveBaseModelCheckpoint(
            filepath=os.path.join(config["save_directory"], "best_model.keras"),
            monitor="val_loss",
            save_best_only=ckpt_cfg["save_best_only"],
            verbose=1
        )
    ]

    # callbacks = [
    #     # ReduceLROnPlateau(
    #     #     monitor=reduce_cfg["monitor"],
    #     #     factor=reduce_cfg["factor"],
    #     #     patience=reduce_cfg["patience"],
    #     #     min_lr=reduce_cfg["min_lr"],
    #     #     verbose=1
    #     # ),
    #     EarlyStopping(
    #         monitor=early_cfg["monitor"],
    #         patience=early_cfg["patience"],
    #         min_delta=reduce_cfg["min_delta"],
    #         restore_best_weights=early_cfg["restore_best_weights"],
    #         verbose=1
    #     ),
    #     ModelCheckpoint(
    #         filepath=os.path.join(config["save_directory"], "best_model.keras"),
    #         monitor="val_loss",
    #         save_best_only=ckpt_cfg["save_best_only"],
    #         verbose=1
    #     )
    # ]

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
    enc = find_encoder(model)

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
    # val_gen = AdaptedGenerator(val_gen, adapter)

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
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=mode_config.get("default_epochs", config.get("epochs", 20)),
        steps_per_epoch=config["steps_per_epoch"],
        callbacks=callbacks,
        verbose=1,
    )

    model.evaluate(val_gen, steps=1)


    # ---------------------------------- SAVE MODELs ---------------------------------
    head_saved = False
    encoder_saved = False

    if config.get("save_head_model", True):
        print(f"💾 Saving head model to {head_path}")
        head_model = model.base_model
        # head_model = model.base_model if isinstance(model, DualOptimizerModel) else model
        head_model.save(head_path)
        head_saved = True
    if config.get("save_encoder_model", True):
        save_encoder(model.base_model, encoder_path)
        # save_encoder(model.base_model if isinstance(model, DualOptimizerModel) else model, encoder_path)
        encoder_saved = True

    info = {
        "mode": mode,
        "submode": config.get("submode"),
        "head_path": head_path if head_saved else None,
        "encoder_path": encoder_path if encoder_saved else None,
        "encoder_lr": encoder_lr,
        "head_lr": head_lr,
        "history": getattr(history, "history", None),
        "start_time": start_time,
        "end_time": time.time(),
    }
    
    if return_info:
        return model, info
    return model

def get_encoder_from_model(model):
    """Return the shared CardStateEncoder if present."""
    # if isinstance(model, DualOptimizerModel):
    #     model = model.base_model
    
    for layer in model.layers:
        if isinstance(layer, CardStateEncoder):
           return layer
    for sublayer in getattr(layer, "layers", []):
        if isinstance(sublayer, CardStateEncoder):
           return sublayer
    return None

def find_encoder(model):
    # if isinstance(model, DualOptimizerModel):
    #     model = model.base_model

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
    # if isinstance(model, DualOptimizerModel):
    #     model = model.base_model

    encoder = get_encoder_from_model(model)
    if encoder:
        print(f"💾 Saving shared encoder weights to {path}")
        dummy = tf.zeros((1, 14, 4, 1))
        # dummy = tf.zeros((1, 13, 4, 2))
        encoder(dummy, training=False)
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

def migrate_dual_to_base(path: str, out_path: str = None, overwrite: bool = False):
	"""Attempt to load a DualOptimizerModel wrapper and save its base_model to disk.

	Notes:
	- This requires that the saved file can be deserialized into a DualOptimizerModel
	  (i.e. the environment contains compatible class definitions).
	- If overwrite=True and migration succeeds, the original file will be replaced.
	"""
	custom = get_custom_objects()
	custom["DualOptimizerModel"] = DualOptimizerModel
	m = tf.keras.models.load_model(path, custom_objects=custom, compile=False, safe_mode=False)

	if not isinstance(m, DualOptimizerModel) or getattr(m, "base_model", None) is None:
		raise RuntimeError("Loaded object is not a DualOptimizerModel with a base_model; cannot migrate.")

	base = m.base_model
	out_path = out_path or (path if overwrite else f"{path}.base.keras")
	base.save(out_path)
	return out_path


