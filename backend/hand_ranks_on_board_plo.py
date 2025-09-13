import numpy as np
from itertools import combinations
from collections import defaultdict
import time
from typing import List, Dict, Tuple

from db_plo import DB_PLO, open_db
from db_plo_operations_treys import cards_to_bitmask

from card import RANK_ORDER, SUIT_ORDER, RANKS, SUITS


def rank_all_hands_all_runouts(db, board, hands_to_rank, debug = False):
    """
    Rank all possible PLO hands on all possible runouts from a given board state.
    
    Args:
        db: Database connection object with get_evaluations_for_hands_and_boards_bm()
        board: Partial board (3, 4, or 5 cards)
        hands_to_rank: List of 4-card PLO hands to get percentiles for
        debug: Whether to print debug information
    
    Returns:
        List of numpy arrays, one per hand in hands_to_rank. Each array contains percentiles
        for that hand across all possible runouts (shape: [num_runouts])
    """
    
    if len(board) not in [3, 4, 5]:
        raise ValueError("Board must be 3, 4, or 5 cards")
    
    start_time = time.time()
    
    # Get all cards not on the board
    all_cards = [rank + suit for rank in RANKS for suit in SUITS]
    available_cards = [card for card in all_cards if card not in board]
    
    # Generate all possible complete 5-card boards
    complete_boards = generate_all_complete_boards(board, set())
    
    if debug:
        print(f"Board: {board} ({len(board)} cards)")
        print(f"Available cards: {len(available_cards)}")
        print(f"Total possible runouts: {len(complete_boards):,}")
    
    # Generate ALL possible 4-card PLO hands from available cards
    all_possible_hands = list(combinations(available_cards, 4))
    
    if debug:
        print(f"Total possible 4-card hands: {len(all_possible_hands):,}")
    
    # Pre-compute all 2-card combinations and their bitmasks
    hand_combo_masks = {}  # hand_idx -> [list of 2-card combo masks]
    all_unique_hand_masks = set()
    
    for hand_idx, four_card_hand in enumerate(all_possible_hands):
        combo_masks = []
        for two_card_combo in combinations(four_card_hand, 2):
            combo_mask = cards_to_bitmask(list(two_card_combo))
            combo_masks.append(combo_mask)
            all_unique_hand_masks.add(combo_mask)
        hand_combo_masks[hand_idx] = combo_masks
    
    # Pre-compute all 3-card board subsets and their bitmasks
    board_subset_masks = {}  # board_idx -> [list of 3-card subset masks]
    all_unique_board_masks = set()
    
    for board_idx, complete_board in enumerate(complete_boards):
        subset_masks = []
        for three_card_subset in combinations(complete_board, 3):
            sorted_subset = tuple(sorted(three_card_subset, key=card_sort_key))
            subset_mask = cards_to_bitmask(list(sorted_subset))
            subset_masks.append(subset_mask)
            all_unique_board_masks.add(subset_mask)
        board_subset_masks[board_idx] = subset_masks
    
    if debug:
        print(f"Unique 2-card hand combinations: {len(all_unique_hand_masks):,}")
        print(f"Unique 3-card board subsets: {len(all_unique_board_masks):,}")
        print(f"Setup took: {time.time() - start_time:.2f}s")
    
    # Fetch ALL evaluations from database in one massive query
    db_start = time.time()
    evaluations = db.get_evaluations_for_hands_and_boards_bm(
        list(all_unique_hand_masks), 
        list(all_unique_board_masks)
    )
    
    if debug:
        print(f"Database query took: {time.time() - db_start:.2f}s")
        print(f"Retrieved {len(evaluations):,} evaluations")
    
    # Initialize the massive results array
    num_hands = len(all_possible_hands)
    num_runouts = len(complete_boards)
    
    # We'll compute rankings one runout at a time to manage memory
    if debug:
        print(f"Computing rankings for {num_hands:,} hands across {num_runouts:,} runouts...")
    
    # Find indices of our target hands
    target_hand_indices = []
    for target_hand in hands_to_rank:
        if len(target_hand) != 4:
            target_hand_indices.append(None)
            continue
        
        # Check if this hand conflicts with the board
        if any(card in board for card in target_hand):
            target_hand_indices.append(None)
            if debug:
                print(f"Hand {target_hand} conflicts with board {board}")
            continue
        
        # Find this hand in our all_possible_hands list
        target_hand_sorted = tuple(sorted(target_hand, key=card_sort_key))
        target_hand_idx = None
        
        for idx, possible_hand in enumerate(all_possible_hands):
            possible_hand_sorted = tuple(sorted(possible_hand, key=card_sort_key))
            if possible_hand_sorted == target_hand_sorted:
                target_hand_idx = idx
                break
        
        target_hand_indices.append(target_hand_idx)
        if debug and target_hand_idx is not None:
            print(f"Hand {target_hand} found at index {target_hand_idx}")
        elif debug:
            print(f"Hand {target_hand} not found in possible hands")
    
    # Initialize result arrays for target hands
    result_arrays = []
    for target_idx in target_hand_indices:
        if target_idx is not None:
            result_arrays.append(np.full(num_runouts, np.nan))
        else:
            result_arrays.append(None)
    
    # Process each runout
    ranking_start = time.time()
    valid_runouts = 0
    
    for runout_idx, complete_board in enumerate(complete_boards):
        if debug and runout_idx % 1000 == 0:
            elapsed = time.time() - ranking_start
            rate = runout_idx / elapsed if elapsed > 0 else 0
            remaining = (num_runouts - runout_idx) / rate if rate > 0 else 0
            print(f"Processed {runout_idx:,}/{num_runouts:,} runouts ({runout_idx/num_runouts*100:.1f}%) "
                  f"- {rate:.0f} runouts/sec - ETA: {remaining:.0f}s")
        
        # Calculate best hand value for each possible hand on this runout
        hand_values = []
        
        for hand_idx in range(num_hands):
            best_value = float('inf')  # Lower is better
            
            # Check all 2-card combinations from this hand against all 3-card subsets of this board
            for hand_combo_mask in hand_combo_masks[hand_idx]:
                for board_subset_mask in board_subset_masks[runout_idx]:
                    if (hand_combo_mask, board_subset_mask) in evaluations:
                        value = evaluations[(hand_combo_mask, board_subset_mask)][0]  # high_value
                        best_value = min(best_value, value)
            
            # Only include hands that have valid evaluations
            if best_value != float('inf'):
                hand_values.append((hand_idx, best_value))
        
        if not hand_values:
            continue  # Skip runouts with no valid hands
        
        valid_runouts += 1
        
        # Rank all hands for this runout
        hand_rankings = rank_hands_for_board_universal(hand_values)
        
        # Extract percentiles for our target hands
        for target_result_idx, target_hand_idx in enumerate(target_hand_indices):
            if target_hand_idx is None or result_arrays[target_result_idx] is None:
                continue
            
            # Find this hand's ranking
            for hand_id, value, rank_min, rank_max, rank_avg, rank_dense in hand_rankings:
                if hand_id == target_hand_idx:
                    result_arrays[target_result_idx][runout_idx] = rank_avg
                    break
    
    if debug:
        total_time = time.time() - start_time
        print(f"\nRanking complete!")
        print(f"Valid runouts processed: {valid_runouts:,}/{num_runouts:,}")
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Average time per runout: {total_time/num_runouts*1000:.2f}ms")
        
        # Print summary stats for each target hand
        for i, (hand, result_array) in enumerate(zip(hands_to_rank, result_arrays)):
            if result_array is not None:
                valid_percentiles = result_array[~np.isnan(result_array)]
                if len(valid_percentiles) > 0:
                    print(f"\nHand {i} ({hand}):")
                    print(f"  Valid runouts: {len(valid_percentiles):,}")
                    print(f"  Mean percentile: {np.mean(valid_percentiles):.4f}")
                    print(f"  Min percentile: {np.min(valid_percentiles):.4f}")
                    print(f"  Max percentile: {np.max(valid_percentiles):.4f}")
                    print(f"  Std percentile: {np.std(valid_percentiles):.4f}")
    
    return result_arrays


def generate_all_complete_boards(partial_board, dealt_cards):
    """
    Generate all possible complete 5-card boards from a partial board.
    
    Args:
        partial_board: List of 0-5 cards already on the board
        dealt_cards: Set of cards already dealt (will be ignored for PLO ranking)
    
    Returns:
        List of all possible complete 5-card boards
    """
    
    if len(partial_board) > 5:
        raise ValueError("Partial board cannot have more than 5 cards")
    
    if len(partial_board) == 5:
        return [partial_board]
    
    # Get all available cards (not on the board, ignore dealt_cards for universal ranking)
    all_cards = [rank + suit for rank in RANKS for suit in SUITS]
    available_cards = [card for card in all_cards if card not in partial_board]
    
    cards_needed = 5 - len(partial_board)
    
    # Generate all possible ways to complete the board
    complete_boards = []
    for additional_cards in combinations(available_cards, cards_needed):
        complete_board = partial_board + list(additional_cards)
        complete_boards.append(complete_board)
    
    return complete_boards


def analyze_hand_stability_across_runouts(percentile_arrays, hand_names=None):
    """
    Analyze how stable each hand's performance is across different runouts.
    
    Args:
        percentile_arrays: List of numpy arrays from rank_all_hands_all_runouts()
        hand_names: Optional list of hand names for labeling
    
    Returns:
        Dict with stability analysis for each hand
    """
    
    if hand_names is None:
        hand_names = [f"Hand {i}" for i in range(len(percentile_arrays))]
    
    results = {}
    
    for i, (percentiles, name) in enumerate(zip(percentile_arrays, hand_names)):
        if percentiles is None:
            results[name] = None
            continue
        
        valid_percentiles = percentiles[~np.isnan(percentiles)]
        
        if len(valid_percentiles) == 0:
            results[name] = None
            continue
        
        # Calculate stability metrics
        results[name] = {
            'mean': np.mean(valid_percentiles),
            'median': np.median(valid_percentiles),
            'std': np.std(valid_percentiles),
            'min': np.min(valid_percentiles),
            'max': np.max(valid_percentiles),
            'range': np.max(valid_percentiles) - np.min(valid_percentiles),
            'q25': np.percentile(valid_percentiles, 25),
            'q75': np.percentile(valid_percentiles, 75),
            'iqr': np.percentile(valid_percentiles, 75) - np.percentile(valid_percentiles, 25),
            'coefficient_of_variation': np.std(valid_percentiles) / np.mean(valid_percentiles),
            'num_valid_runouts': len(valid_percentiles)
        }
    
    return results


def plot_hand_runout_distributions(percentile_arrays, hand_names = None, 
                                  title = None):
    """
    Plot distributions of hand percentiles across all runouts.
    
    Args:
        percentile_arrays: List of numpy arrays from rank_all_hands_all_runouts()
        hand_names: Optional list of hand names for labeling
        title: Optional plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if hand_names is None:
        hand_names = [f"Hand {i}" for i in range(len(percentile_arrays))]
    
    # Filter out None arrays
    valid_arrays = []
    valid_names = []
    for arr, name in zip(percentile_arrays, hand_names):
        if arr is not None:
            valid_percentiles = arr[~np.isnan(arr)]
            if len(valid_percentiles) > 0:
                valid_arrays.append(valid_percentiles)
                valid_names.append(name)
    
    if not valid_arrays:
        print("No valid data to plot")
        return
    
    # Create subplots
    fig, axes = plt.subplots(len(valid_arrays), 1, figsize=(10, 3 * len(valid_arrays)), sharex=True)
    if len(valid_arrays) == 1:
        axes = [axes]
    
    for i, (percentiles, name) in enumerate(zip(valid_arrays, valid_names)):
        ax = axes[i]
        
        # Create histogram
        ax.hist(percentiles * 100, bins=50, alpha=0.7, edgecolor='black', density=True)
        
        # Add mean line
        mean_pct = np.mean(percentiles) * 100
        ax.axvline(mean_pct, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_pct:.1f}%')
        
        # Add quartile lines
        q25 = np.percentile(percentiles, 25) * 100
        q75 = np.percentile(percentiles, 75) * 100
        ax.axvline(q25, color='orange', linestyle=':', alpha=0.7, label=f'Q25: {q25:.1f}%')
        ax.axvline(q75, color='orange', linestyle=':', alpha=0.7, label=f'Q75: {q75:.1f}%')
        
        ax.set_title(f'{name} - Percentile Distribution Across All Runouts')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    axes[-1].set_xlabel('Percentile (%)')
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.show()


def rank_all_hands_on_board(db, board: List[str], hands_to_rank: List[List[str]], debug: bool = False) -> List[float]:
    """
    Rank all possible PLO hands on a given 3-card flop and return percentiles for specific hands.
    
    Args:
        db: Database connection object with get_evaluations_for_hands_and_boards_bm()
        board: 3-card flop
        hands_to_rank: List of 4-card PLO hands to get percentiles for
        debug: Whether to print debug information
    
    Returns:
        List of percentiles (0.0-1.0) for each hand in hands_to_rank, where 0.0 = worst, 1.0 = best
    """
    from itertools import combinations
    from collections import defaultdict
    import time
    
    if len(board) != 3:
        raise ValueError("Board must be exactly 3 cards (flop)")
    
    start_time = time.time()
    
    # Get all cards not on the board
    all_cards = generate_deck()  # Assuming you have this function
    available_cards = [card for card in all_cards if card not in board]
    
    if debug:
        print(f"Available cards for hands: {len(available_cards)}")
    
    # Generate ALL possible 4-card PLO hands from available cards
    all_possible_hands = list(combinations(available_cards, 4))
    
    if debug:
        print(f"Total possible 4-card hands: {len(all_possible_hands):,}")
    
    # Convert board to bitmask (this is our fixed 3-card subset)
    board_mask = cards_to_bitmask(board)
    
    # Generate all 2-card combinations from all possible hands and convert to bitmasks
    all_hand_combos = []
    hand_combo_to_full_hand = {}  # Map from 2-card combo mask to original 4-card hand index
    
    for hand_idx, four_card_hand in enumerate(all_possible_hands):
        for two_card_combo in combinations(four_card_hand, 2):
            combo_mask = cards_to_bitmask(list(two_card_combo))
            all_hand_combos.append(combo_mask)
            if combo_mask not in hand_combo_to_full_hand:
                hand_combo_to_full_hand[combo_mask] = []
            hand_combo_to_full_hand[combo_mask].append(hand_idx)
    
    if debug:
        print(f"Total 2-card combinations: {len(all_hand_combos):,}")
        print(f"Unique 2-card combinations: {len(set(all_hand_combos)):,}")
    
    # Get unique hand masks for database query
    unique_hand_masks = list(set(all_hand_combos))
    
    # Fetch all evaluations from database in one query
    evaluations = db.get_evaluations_for_hands_and_boards_bm(unique_hand_masks, [board_mask])
    
    if debug:
        print(f"Database lookup took: {time.time() - start_time:.2f}s")
        print(f"Retrieved {len(evaluations):,} evaluations")
    
    # Calculate best hand value for each possible 4-card hand
    hand_values = []
    
    for hand_idx, four_card_hand in enumerate(all_possible_hands):
        best_value = float('inf')  # Lower is better
        
        # Check all 2-card combinations from this hand
        for two_card_combo in combinations(four_card_hand, 2):
            combo_mask = cards_to_bitmask(list(two_card_combo))
            
            if (combo_mask, board_mask) in evaluations:
                value = evaluations[(combo_mask, board_mask)][0]  # high_value
                best_value = min(best_value, value)
        
        # Only include hands that have valid evaluations
        if best_value != float('inf'):
            hand_values.append((hand_idx, best_value))
        elif debug and len(hand_values) < 10:
            print(f"Hand {hand_idx} ({four_card_hand}) has no valid evaluation")
    
    if debug:
        print(f"Valid hands for ranking: {len(hand_values):,}")
        print(f"Hand evaluation took: {time.time() - start_time:.2f}s")
    
    # Rank all hands using the same logic as Hold'em
    hand_rankings = rank_hands_for_board_universal(hand_values)
    
    if debug:
        print(f"Ranking took: {time.time() - start_time:.2f}s")
    
    # Now find the percentiles for our specific hands to rank
    percentiles = []
    
    for target_hand in hands_to_rank:
        if len(target_hand) != 4:
            percentiles.append(None)
            continue
        
        # Check if this hand conflicts with the board
        if any(card in board for card in target_hand):
            percentiles.append(None)
            if debug:
                print(f"Hand {target_hand} conflicts with board {board}")
            continue
        
        # Find this hand in our all_possible_hands list
        target_hand_sorted = tuple(sorted(target_hand, key=card_sort_key))
        target_hand_idx = None
        
        for idx, possible_hand in enumerate(all_possible_hands):
            possible_hand_sorted = tuple(sorted(possible_hand, key=card_sort_key))
            if possible_hand_sorted == target_hand_sorted:
                target_hand_idx = idx
                break
        
        if target_hand_idx is None:
            percentiles.append(None)
            if debug:
                print(f"Could not find hand {target_hand} in possible hands")
            continue
        
        # Find the ranking for this hand
        target_percentile = None
        for hand_id, value, rank_min, rank_max, rank_avg, rank_dense in hand_rankings:
            if hand_id == target_hand_idx:
                target_percentile = rank_avg  # Use average percentile
                break
        
        percentiles.append(target_percentile)
        
        if debug:
            print(f"Hand {target_hand}: percentile = {target_percentile}")
    
    if debug:
        print(f"Total execution time: {time.time() - start_time:.2f}s")
    
    return percentiles


def rank_hands_for_board_universal(hand_values):
    """
    Universal ranking function for any list of (hand_id, value) pairs.
    
    Args:
        hand_values: List of tuples of (hand_id, value)

    Returns:
        List of tuples of (hand_id, value, rank_min, rank_max, rank_avg, rank_dense)
    """
    from collections import defaultdict
    
    if not hand_values:
        return []
    
    # Sort by value (lower is better)
    hand_values.sort(key=lambda x: x[1])
    
    # Group by value
    value_to_indices = defaultdict(list)
    for idx, (hand_id, value) in enumerate(hand_values):
        value_to_indices[value].append(idx)

    sorted_values = sorted(value_to_indices.keys())
    hand_rankings = [None] * len(hand_values)

    num_unique_values = len(sorted_values)
    num_total_hands = len(hand_values)

    denom_non_dense = num_total_hands - 1 if num_total_hands > 1 else 1
    denom_dense = num_unique_values - 1 if num_unique_values > 1 else 1

    pos = 0
    for rank_index, value in enumerate(sorted_values):
        indices = value_to_indices[value]

        min_rank_val = pos
        max_rank_val = pos + len(indices) - 1
        avg_rank_val = pos + (len(indices) - 1) / 2
        dense_rank_val = rank_index

        min_percentile = min_rank_val / denom_non_dense
        max_percentile = max_rank_val / denom_non_dense
        avg_percentile = avg_rank_val / denom_non_dense
        dense_percentile = dense_rank_val / denom_dense

        for idx in indices:
            hand_id, hand_value = hand_values[idx]
            hand_rankings[idx] = (hand_id, hand_value, min_percentile, max_percentile, avg_percentile, dense_percentile)

        pos += len(indices)

    return hand_rankings


def generate_deck():
    """Generate a standard 52-card deck."""
    suits = ['h', 'd', 'c', 's']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    return [rank + suit for suit in suits for rank in ranks]


def analyze_board_texture_plo(db, board: List[str], sample_hands: List[List[str]] = None, debug: bool = False):
    """
    Analyze the texture of a PLO flop by examining hand strength distributions.
    
    Args:
        db: Database connection object
        board: 3-card flop
        sample_hands: Optional list of specific hands to analyze
        debug: Whether to print debug information
    
    Returns:
        Dict with board analysis including equity distributions
    """
    import random
    
    if sample_hands is None:
        # Generate a random sample of hands for analysis
        all_cards = generate_deck()
        available_cards = [card for card in all_cards if card not in board]
        
        # Sample some random hands
        sample_hands = []
        for _ in range(20):  # Sample 20 random hands
            hand = random.sample(available_cards, 4)
            # Make sure this hand doesn't conflict with already sampled hands
            hand_cards = set(hand)
            conflict = False
            for existing_hand in sample_hands:
                if hand_cards & set(existing_hand):
                    conflict = True
                    break
            if not conflict:
                sample_hands.append(hand)
                # Remove these cards from available pool for next hand
                available_cards = [c for c in available_cards if c not in hand]
    
    # Get percentiles for sample hands
    percentiles = rank_all_hands_on_board(db, board, sample_hands, debug=debug)
    
    # Analyze the distribution
    valid_percentiles = [p for p in percentiles if p is not None]
    
    result = {
        'board': board,
        'sample_hands': sample_hands,
        'percentiles': percentiles,
        'valid_percentiles': valid_percentiles,
        'num_hands_analyzed': len(valid_percentiles),
    }
    
    if valid_percentiles:
        result.update({
            'mean_percentile': sum(valid_percentiles) / len(valid_percentiles),
            'min_percentile': min(valid_percentiles),
            'max_percentile': max(valid_percentiles),
            'percentile_range': max(valid_percentiles) - min(valid_percentiles)
        })
    
    return result


def plot_board_analysis(analysis_result):
    """Plot the results of a board texture analysis."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    percentiles = analysis_result['valid_percentiles']
    board = analysis_result['board']
    
    if not percentiles:
        print("No valid percentiles to plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    plt.hist(percentiles, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(analysis_result['mean_percentile'], color='red', linestyle='--', 
                label=f"Mean: {analysis_result['mean_percentile']:.3f}")
    
    plt.title(f"Hand Strength Distribution on {board}")
    plt.xlabel("Percentile (0=worst, 1=best)")
    plt.ylabel("Number of Hands")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()



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

    print("Comprehensive PLO Hand Ranking Example")
    print("=" * 50)
    
    # Example for different board states
    examples = [
        {
            'board': ["As", "Jc", "4s"],
            'description': "Dry flop"
        },
        {
            'board': ["As", "Jc", "4s", "Kh"], 
            'description': "Flop + turn"
        },
        {
            'board': ["7h", "8s", "9c"],
            'description': "Connected flop"
        }
    ]
    
    hands_to_analyze = [
        ["Ah", "9d", "8h", "6d"],  # Top pair + straight draw
        ["Kc", "Qc", "Jd", "9s"],  # Straight + flush draws
        ["6h", "5h", "4h", "3h"]   # Low straight + flush draw
    ]
    
    for example in examples:
        print(f"\nBoard: {example['board']} ({example['description']})")
        print(f"Hands to analyze: {hands_to_analyze}")
        print(f"This will return {len(hands_to_analyze)} arrays, each with ~{'1000' if len(example['board']) == 3 else '50' if len(example['board']) == 4 else '1'} percentiles")
        print("Each percentile shows how that hand ranks against ALL possible hands on that runout")
    
    print("\n" + "=" * 50)
    print("Sample output interpretation:")
    print("If Hand 1 has percentiles [0.75, 0.82, 0.71, ...] it means:")
    print("- On runout 1: better than 75% of all possible hands")  
    print("- On runout 2: better than 82% of all possible hands")
    print("- On runout 3: better than 71% of all possible hands")
    print("- etc.")


    result_arrays = rank_all_hands_all_runouts(db, ["As", "Jc", "4s", "2c"], hands_to_analyze, debug = False)
    print(result_arrays)

    plot_hand_runout_distributions(result_arrays, hands_to_analyze)


    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
 
    # Close the DB connection
    db.close()


if __name__ == "__main__":
    main()

