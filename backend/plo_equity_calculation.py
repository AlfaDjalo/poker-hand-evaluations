from itertools import combinations
import math
import time

from db_plo import DB_PLO, open_db
from card import RANK_ORDER, SUIT_ORDER

# --- Helpers you already have/need ---

RANKS = list("AKQJT98765432")   # match your order
SUITS = list("shdc")            # match your order

def full_deck():
    return [r+s for r in RANKS for s in SUITS]

def canonical_2card(c1, c2):
    return "".join(sorted((c1, c2), key=card_sort_key))

def canonical_3card(b3):
    return "".join(sorted(b3, key=card_sort_key))

def three_card_subsets(board5):
    # returns the 10 three-card subsets of a 5-card board, canonicalized
    return [canonical_3card(t) for t in combinations(board5, 3)]

# --- Efficient fetch of needed evals ---

def get_evaluations_for_hands_and_boards(db, hand_ids, board_ids, batch_size=50000):
    """
    Returns dict {(hand_id, board_id): (high_hand_value, rank_dense)}.
    Batches IN-clauses to avoid parameter explosions.
    """
    if not hand_ids or not board_ids:
        return {}

    evals = {}
    hand_ids = list(dict.fromkeys(hand_ids))
    board_ids = list(dict.fromkeys(board_ids))

    # batch over boards (usually more numerous)
    b_batches = math.ceil(len(board_ids)/batch_size)
    for bi in range(b_batches):
        b_slice = board_ids[bi*batch_size:(bi+1)*batch_size]
        ph = ",".join(["%s"] * len(hand_ids))
        pb = ",".join(["%s"] * len(b_slice))
        q = f"""
            SELECT hand_id, board_id, hand_value, rank_dense
            FROM plo_evaluations
            WHERE hand_id IN ({ph})
              AND board_id IN ({pb})
        """
        db.cursor.execute(q, hand_ids + b_slice)
        for h_id, b_id, hv, rd in db.cursor.fetchall():
            evals[(h_id, b_id)] = (hv, rd)

    return evals

# --- Main exhaustive equity ---

def calculate_equity_for_multiple_hands_exhaustive(db, hand_list, board=None,
                                                   value_selector="high_hand_value",
                                                   batch_new_boards=50000):
    """
    Exhaustive equity across COMPLETE 5-card runouts consistent with known cards.
    - hand_list: list of PLO hands, each a list[str] of 4 cards, e.g. ["As","Kh","Jc","3c"]
    - board: None / 3-card flop / 4-card turn / 5-card river
    - value_selector: "high_hand_value" (maximize). If your metric is inverted, flip comparator below.
    - batch_new_boards: how many *new* 3-card board IDs to fetch at a time for caching.

    Returns: list of equities (floats summing to 1.0), one per hand in hand_list.
    """

    board = board or []
    # 1) Build remaining deck (exclude all known cards)
    used = set(c for hand in hand_list for c in hand)
    used.update(board)
    deck = [c for c in full_deck() if c not in used]

    # 2) Enumerate COMPLETE 5-card runouts
    if len(board) == 0:
        runouts = combinations(deck, 5)
    elif len(board) == 3:
        runouts = (tuple(board) + rc for rc in combinations(deck, 2))
    elif len(board) == 4:
        runouts = (tuple(board) + (c,) for c in deck)
    elif len(board) == 5:
        runouts = [tuple(board)]
    else:
        raise ValueError("board must be None, or have length 3, 4, or 5.")

    # 3) ID maps
    hand_id_map  = db.get_hand_ids()   # 2-card → id
    board_id_map = db.get_board_ids()  # 3-card → id (size ≈ 22100, fine to keep in RAM)

    # 4) Precompute each player's 6 two-card IDs
    players_hids = []
    for hand in hand_list:
        hids = []
        for c1, c2 in combinations(hand, 2):
            key = canonical_2card(c1, c2)
            if key in hand_id_map:
                hids.append(hand_id_map[key])
        if not hids:
            # shouldn't happen if DB complete
            hids = []
        players_hids.append(hids)
    
    # 5) On-demand cache for (hand_id, board3_id) → (value, rank)
    eval_cache = {}
    missing_board3_ids = set()

    # choose comparator: by default we MAXIMIZE high_hand_value
    # If your metric is reversed, set `better = min` and init to +inf
    def better_val_init():
        return +math.inf
    def better(a, b):
        return a if a <= b else b

    # 6) Tally equities by counting wins/splits over all runouts
    wins = [0.0] * len(hand_list)
    total_runouts = 0

    def ensure_loaded(b3_ids_needed):
        """Fetch missing (hand_id, b3_id) rows in batches and fill eval_cache."""
        nonlocal missing_board3_ids
        # collect NEW board IDs
        new_ids = [b for b in b3_ids_needed if b not in missing_board3_ids]
        if not new_ids:
            return
        # fetch for all player hand_ids × new board IDs
        all_hand_ids = [hid for ids in players_hids for hid in ids]
        if not all_hand_ids or not new_ids:
            return
        fetched = get_evaluations_for_hands_and_boards(db, all_hand_ids, new_ids)
        eval_cache.update(fetched)
        # mark as seen to avoid re-fetch attempts
        for b in new_ids:
            missing_board3_ids.add(b)

    batch_pending = set()  # accumulate unseen b3_ids before fetching

    for board5 in runouts:
        total_runouts += 1
        # Get the 10 three-card subset IDs for this 5-card board
        b3_keys = three_card_subsets(board5)
        b3_ids = []
        for k in b3_keys:
            b_id = board_id_map.get(k)
            if b_id is not None:
                b3_ids.append(b_id)
                if b_id not in missing_board3_ids and all(
                    (hid, b_id) in eval_cache for hids in players_hids for hid in hids
                ) is False:
                    batch_pending.add(b_id)

        # Batch load missing evals periodically
        if len(batch_pending) >= batch_new_boards:
            ensure_loaded(list(batch_pending))
            batch_pending.clear()

        # Compute each player's BEST value on this board5
        best_vals = []
        for hids in players_hids:
            best = better_val_init()
            for b_id in b3_ids:
                for hid in hids:
                    tup = eval_cache.get((hid, b_id))
                    if tup is None:
                        continue
                    val = tup[0] if value_selector == "high_hand_value" else tup[1]
                    best = better(best, val)
            best_vals.append(best)

        # Determine winners (handle splits)
        max_val = max(best_vals)
        winners = [i for i, v in enumerate(best_vals) if v == max_val]
        share = 1.0 / len(winners)
        for i in winners:
            wins[i] += share

    # Flush any last pending (not strictly needed at the end)
    if batch_pending:
        ensure_loaded(list(batch_pending))

    if total_runouts == 0:
        return [0.0] * len(hand_list)

    equities = [w / total_runouts for w in wins]
    return equities

def card_sort_key(card_str: str):
    """
    Sorting key for cards represented as 2-character strings like "As", "Qh", "Td".
    """
    rank = card_str[0]
    suit = card_str[1]
    return (RANK_ORDER[rank], SUIT_ORDER[suit])


def main():

    # Initialize DB connection
    db = open_db()
 
    start_time = time.time()
    equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]])
    print(equities)
    equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]], ["Tc", "7d", "7c"])
    print(equities)
    equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]], ["Tc", "7d", "7c", "2h"])
    print(equities)
    equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]], ["Tc", "7d", "7c", "2h", "4d"])
    print(equities)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
 
    # Close the DB connection
    db.close()


if __name__ == "__main__":
    main()
