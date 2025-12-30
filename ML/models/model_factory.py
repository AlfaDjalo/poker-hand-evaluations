import tensorflow as tf
from tensorflow.keras import layers, Model
from .encoders import *

"""
Model factory for assembling end-to-end Keras models from modular components.

This module builds a complete model by wiring together:
- one or more Keras Input tensors (single or multi-input modes)
- a shared CardStateEncoder
- an InputProcessor that maps raw inputs to encoder inputs
- a task-specific head that consumes encoder embeddings

The factory is intentionally generic and supports multiple model modes,
including:
- single-input classification/regression
- multi-input (e.g. hand / board) models
- pairwise comparison models
- embedding-based objectives

All structural variation is driven by `mode_config`.
"""

def build_model(config, mode_config):
    """
    Assemble a complete Keras model from encoder, processor, and head components.

    This factory function constructs models in a highly configurable way,
    supporting both single-input and multi-input architectures. The exact
    wiring is controlled by `mode_config`, allowing the same encoder and
    training pipeline to be reused across different problem formulations.

    High-level flow:
        1. Create one or more Keras Input tensors based on `mode_config`
        2. Instantiate a shared CardStateEncoder
        3. Apply an InputProcessor to transform raw inputs into encoder inputs
        4. Encode inputs into embeddings
        5. Apply a task-specific head to produce final outputs

    Args:
        config (dict):
            Global model or experiment configuration. This is expected to
            contain encoder-related settings (e.g. embedding_dim) and may
            include a `submode` key to select among multiple heads.

        mode_config (dict):
            Mode-specific configuration defining:
            - input structure (single vs multiple inputs)
            - input processor class
            - available head models

            Expected keys include:
            - "input_shape" OR "inputs"
            - "input_processor"
            - "head_model"

    Returns:
        tf.keras.Model:
            A compiled (but untrained) Keras model with inputs and outputs
            defined according to the selected mode.
    """
    # Determine whether this is a single-input or multi-input model
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

    # Shared encoder instance used across all inputs
    encoder_config = get_encoder_config(config)
    encoder = CardStateEncoder(encoder_config)

    # InputProcessor defines how raw model inputs are mapped
    # to encoder inputs (e.g. hand-only, hand+board, pairwise A/B).
    processor_cls = mode_config["input_processor"]
    processor = processor_cls(encoder)
    
    embeddings = processor(inputs)

    # Select head based on submode (e.g. combined, separate, pairwise)
    submode = config.get("submode", "combined")
    head_cls = mode_config["head_model"][submode]
    head = head_cls()
    
    outputs = head(embeddings)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model    