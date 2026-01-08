from bindings.holdem_wrapper import evaluate_showdown

def evaluate_hand_cpp(hand_cards, board_cards):
    """
    Evaluate Hold'em showdown using C++ evaluator.
    
    For push-fold, we always have 2 players.
    This function should be called from _resolve_showdown.
    
    Parameters
    ----------
    hand_cards : list[str]
        Two hole cards (e.g., ["As", "Kd"])
    board_cards : list[str]
        Five board cards
        
    Returns
    -------
    int
        Hand rank for comparison (lower rank wins in OMPEval, so we return negative)
    """
    # Note: This is a simplified version for your use case
    # You'll need to adapt this based on how you want to use it
    pass