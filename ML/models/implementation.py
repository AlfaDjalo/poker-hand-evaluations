# Compatibility shim: re-export classes from the split modules.
from .encoders import *   # PokerCNNEncoder, PokerCNNEncoder_old, SuitPermutationLayer, SuitEquivariantLayer
# from .combo import *      # PokerComboModel, ComboConcatLayer, get_encoder_config
# from .heads import *      # PokerValueModel, PokerValueHeads, PokerCategoryModel, PokerCategoryHeads, PairwiseComparisonModel, PairwiseComparisonHeads, WeightedCategoricalCrossentropy
from .grid_value_model import *
from .embedding_value_model import *
from .pairwise_comparison_model import *
from .hand_category_model import *

# Keep a minimal __all__ for explicit exports
__all__ = [
    "CardSetEncoder", "CardStateEncoder", "get_encoder_config",
    # "SuitPermutationLayer","ComboConcatLayer", 
    "HandCategoryHead", "CardStateHandCategoryHead", "build_hand_category_model", "WeightedCategoricalCrossentropy",
    "GridValueHead", "CardStateGridValueHead", "build_grid_value_model",
    "PairwiseComparisonHead", "CardStatePairwiseComparisonHead", "build_pairwise_comparison_model",
    "EmbeddingValueHead", "build_embedding_value_model",    
]

# Note: This file is now a lightweight compatibility layer; implementation lived in encoders.py, combo.py and heads.py