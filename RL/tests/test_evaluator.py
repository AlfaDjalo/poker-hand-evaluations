import pytest

from RL.evaluators.treys_eval import evaluate_hand


def test_aa_beats_kk_holdem():
    """
    AA should always beat KK on the same board in Hold'em.
    """

    # Fixed board: no straight or flush possibilities
    board = ["2h", "7d", "9c", "Jd", "Qs"]

    hand_aa = ["As", "Ah"]
    hand_kk = ["Ks", "Kh"]

    aa_rank = evaluate_hand(board, hand_aa, showdown_style="holdem")
    kk_rank = evaluate_hand(board, hand_kk, showdown_style="holdem")

    # In Treys: lower rank = stronger hand
    assert aa_rank < kk_rank, f"Expected AA ({aa_rank}) to beat KK ({kk_rank})"
