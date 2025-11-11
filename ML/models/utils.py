import tensorflow as tf
from models.implementation import (
    PokerComboModel,
    PokerValueHeads,
    PokerCNNEncoder,
    SuitEquivariantLayer
)

def save_model(model, path):

    model.save(path)

def load_model(path):
    return tf.keras.models.load_model(
        path,
        custom_objects={
            'PokerComboModel': PokerComboModel,
            'PokerValueHeads': PokerValueHeads,
            'PokerCNNEncoder': PokerCNNEncoder,
            'SuitEquivariantLayer': SuitEquivariantLayer
            },
        safe_mode=False
    )