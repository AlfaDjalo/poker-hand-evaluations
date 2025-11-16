import os
import tensorflow as tf

# from ML.models.utils import load_model
from models.implementation import build_value_model, build_pairwise_model
from data.generators import AbsoluteGenerator, PairwiseGenerator, AlternatingGenerator
from models.implementation import (
    PokerComboModel,
    PokerValueModel,
    PokerValueHeads,
    PokerCNNEncoder,
    SuitEquivariantLayer, 
    PairwiseModel
)

def load_value_model(path):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            'PokerComboModel': PokerComboModel,
            "PokerValueModel": PokerValueModel,
            'PokerValueHeads': PokerValueHeads,
            'PokerCNNEncoder': PokerCNNEncoder,
            'SuitEquivariantLayer': SuitEquivariantLayer,
            'PairwiseModel': PairwiseModel
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
        "PairwiseModel": PairwiseModel,
        "SuitEquivariantLayer": SuitEquivariantLayer,
    }

def train_embeddings(config=None):
    """Handles model creation, data, compilation, and training."""
    mode = config["mode"]
    # --- Data ---
    if mode == "absolute_value":
        train_gen = AbsoluteGenerator(config)
    elif mode in ["board", "hand", "mix"]:
        train_gen = PairwiseGenerator(config, mode=mode)
    elif mode == "alternating":
        train_gen = AlternatingGenerator(config)

    # --- Paths ---
    save_dir = config["save_directory"]
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
        load_encoder(model, encoder_path)


    # --- Compile ---
    if mode=="absolute_value":
        loss = ["mse", "mse", "mse"]
        loss_weights = [0.3, 0.3, 0.4]
        # loss = tf.keras.losses.MeanSquaredError()
    else:
        loss = [
            tf.keras.losses.BinaryCrossentropy(from_logits=False),
            tf.keras.losses.BinaryCrossentropy(from_logits=False),
            tf.keras.losses.BinaryCrossentropy(from_logits=False)
            ]
        loss_weights = config.get("pairwise_loss_weights", [0.3, 0.3, 0.4])

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["lr"])
    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)
    # model.compile(optimizer=optimizer, loss=loss)

    print("\n=== FULL MODEL SUMMARY ===")
    model.summary()

    print("\n=== ENCODER SUMMARY ===")
    encoder = get_encoder_from_model(model)
    if encoder is not None:
        encoder.summary()
    else:
        print("No encoder found.")

    # --- Train ---
    model.fit(
        train_gen,
        epochs=config["epochs"],
        steps_per_epoch=config["steps_per_epoch"],
        callbacks=config["callbacks"],
        verbose=1,
    )

    if config["save_model"]:
        if mode == "absolute_value":
            print(f"💾 Saving absolute model to {abs_model_path}")
            model.save(abs_model_path)
        else:
            print(f"💾 Saving pairwise model to {pairwise_model_path}")
            model.save(pairwise_model_path)
        save_encoder(model, encoder_path)
    
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
    if encoder:
        print(f"💾 Saving shared encoder weights to {path}")
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
    # elif not os.path.exists(path):
    #     print(f"⚠️ Encoder file not found at {path}")
    # else:
    #     print("⚠️ No encoder found in model to load into.")
    print("✅ Shared encoder weights loaded successfully.")


