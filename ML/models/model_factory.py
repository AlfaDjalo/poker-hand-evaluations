import tensorflow as tf
from tensorflow.keras import layers, Model
from .encoders import *

def build_model(config, mode_config):
    """
    Builds complete model from mode_config:
        - input grid with two channels for hand/board
        - combined grid created from hand/board grids
        - CardStateEncoder created embeddings for hand/board/combined
        - CombinedInputValueHead creates feed forward network from combined embedding input
        - SeparateInputValueHead creates feed forward network from hand and board embedding inputs
        - value head has sigmoid output representing hand strength from 0 (strongest) to 1 (weakest)

    Args:
        - config (dict): config for models
    """
    if "input_shape" in mode_config:
        # Single input case
        inputs = tf.keras.Input(mode_config["input_shape"], name="model_input")
    elif "inputs" in mode_config:
        # Multiple inputs case
        inputs = []
        for input_cfg in mode_config["inputs"]:
            inp = tf.keras.Input(shape=input_cfg["shape"], name=input_cfg["name"])
            inputs.append(inp)
    else:
        raise ValueError("mode_config must have 'input_shape' or 'inputs' key")

    # inputs = tf.keras.Input(mode_config["input_shape"], name="model_input")

    encoder_config = get_encoder_config(config)
    encoder = CardStateEncoder(encoder_config)

    processor_cls = mode_config["input_processor"]
    processor = processor_cls(encoder)
    
    embeddings = processor(inputs)

    submode = config.get("submode", "combined")
    head_cls = mode_config["head_model"][submode]
    head = head_cls()
    
    outputs = head(embeddings)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model    