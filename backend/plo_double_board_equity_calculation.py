from itertools import combinations
from typing import List, Dict, Tuple
import math
import time

from db_plo import DB_PLO, open_db
from card import RANK_ORDER, SUIT_ORDER, RANKS, SUITS


def calculate_double_board_equity(db, player_hands: List[List[str]], board1: List[str] = None, board2: List[str] = None, debug: bool = False, report: bool = True) -> List[float]:
    """
    Calculate equity for multiple PLO hands in a double board split pot game.
    Each board contributes 50% of the pot to its winner(s).
    
    Args:
        db: Database connection object
        player_hands: List of player hands, each hand is a list of card strings
        board1: Optional list of board1 cards (0-5 cards)
        board2: Optional list of board2 cards (0-5 cards)
        debug: Enable debug output
        report: Print detailed statistics report
    
    Returns:
        List of equity values (as decimals) for each player in order
    """
    if board1 is None:
        board1 = []
    if board2 is None:
        board2 = []
    
    # Pre-flop hack: if no cards dealt on either board, double board equity = single board equity
    if len(board1) == 0 and len(board2) == 0:
        if debug:
            print("Pre-flop scenario: Using single board equity for double board game")
        equity = calculate_equity_for_multiple_hands_exhaustive(db, player_hands, debug=debug)
        
        if report:
            print_double_board_report(player_hands, equity, equity, equity, is_preflop=True)
        
        return equity
    
    # Get all cards in play across both boards and player hands
    dealt_cards = set()
    for hand in player_hands:
        dealt_cards.update(hand)
    dealt_cards.update(board1)
    dealt_cards.update(board2)
    
    # Generate all possible hand combinations for each player
    player_hand_combos = generate_all_player_hand_combinations(player_hands)
    
    # Generate all possible complete board combinations for both boards
    complete_board_pairs = generate_all_double_board_combinations(board1, board2, dealt_cards)
    
    if debug:
        print(f"Total double board combinations: {len(complete_board_pairs)}")
    
    # Get ID mappings from database
    hand_id_map = db.get_hand_ids()
    board_id_map = db.get_board_ids()
    
    # Convert hand combinations to IDs
    all_hand_ids = get_hand_ids_for_all_players(player_hand_combos, hand_id_map)
    
    # Get all unique 3-card subsets from all board pairs
    all_three_card_subsets = set()
    for board_pair in complete_board_pairs:
        board1_complete, board2_complete = board_pair
        # Add 3-card subsets from board1
        for three_card_subset in combinations(board1_complete, 3):
            sorted_subset = tuple(sorted(three_card_subset, key=card_sort_key))
            all_three_card_subsets.add(sorted_subset)
        # Add 3-card subsets from board2
        for three_card_subset in combinations(board2_complete, 3):
            sorted_subset = tuple(sorted(three_card_subset, key=card_sort_key))
            all_three_card_subsets.add(sorted_subset)
    
    # Convert 3-card subsets to board IDs
    subset_board_ids = []
    for subset in all_three_card_subsets:
        board_key = format_board_key(subset)
        if board_key in board_id_map:
            subset_board_ids.append(board_id_map[board_key])
    
    # Get all unique hand IDs for database query
    unique_hand_ids = set()
    for player_hand_ids in all_hand_ids:
        unique_hand_ids.update(player_hand_ids)
    unique_hand_ids = list(unique_hand_ids)
    
    # Fetch all evaluations from database in one query
    evaluations = db.get_evaluations_for_hands_and_boards(unique_hand_ids, subset_board_ids)
    
    if debug:
        print(f"Fetched evaluations for {len(unique_hand_ids)} hands × {len(subset_board_ids)} board subsets")
    
    # Initialize statistics tracking
    num_players = len(player_hands)
    total_equity = [0.0] * num_players
    total_board_pairs = len(complete_board_pairs)
    
    # Board-specific statistics
    board1_wins = [0] * num_players
    board1_chops = [0] * num_players
    board2_wins = [0] * num_players
    board2_chops = [0] * num_players
    
    for pair_idx, (board1_complete, board2_complete) in enumerate(complete_board_pairs):
        if debug and pair_idx < 3:
            print(f"\nEvaluating board pair {pair_idx}:")
            print(f"  Board1: {board1_complete}")
            print(f"  Board2: {board2_complete}")
        
        # Evaluate each board separately
        board1_winners = evaluate_single_complete_board(board1_complete, all_hand_ids, evaluations, board_id_map, debug and pair_idx < 3)
        board2_winners = evaluate_single_complete_board(board2_complete, all_hand_ids, evaluations, board_id_map, debug and pair_idx < 3)
        
        if debug and pair_idx < 3:
            print(f"  Board1 winners: {board1_winners}")
            print(f"  Board2 winners: {board2_winners}")
        
        # Update board-specific statistics
        if board1_winners:
            if len(board1_winners) == 1:
                board1_wins[board1_winners[0]] += 1
            else:
                for winner in board1_winners:
                    board1_chops[winner] += 1
        
        if board2_winners:
            if len(board2_winners) == 1:
                board2_wins[board2_winners[0]] += 1
            else:
                for winner in board2_winners:
                    board2_chops[winner] += 1
        
        # Distribute equity: 50% of pot for each board
        if board1_winners:
            equity_per_board1_winner = 0.5 / len(board1_winners)
            for winner_idx in board1_winners:
                total_equity[winner_idx] += equity_per_board1_winner
        
        if board2_winners:
            equity_per_board2_winner = 0.5 / len(board2_winners)
            for winner_idx in board2_winners:
                total_equity[winner_idx] += equity_per_board2_winner
    
    # Convert to percentages
    final_equity = [equity / total_board_pairs for equity in total_equity]
    board1_stats = [(wins / total_board_pairs * 100, chops / total_board_pairs * 100) 
                    for wins, chops in zip(board1_wins, board1_chops)]
    board2_stats = [(wins / total_board_pairs * 100, chops / total_board_pairs * 100) 
                    for wins, chops in zip(board2_wins, board2_chops)]
    
    if debug:
        print(f"\nTotal board pairs evaluated: {total_board_pairs}")
    
    if report:
        print_double_board_report(player_hands, final_equity, board1_stats, board2_stats)
    
    return final_equity


def print_double_board_report(player_hands: List[List[str]], equity: List[float], 
                             board1_stats, board2_stats, is_preflop: bool = False):
    """
    Print a formatted report of double board equity results.
    """
    print("\n" + "="*60)
    print("DOUBLE BOARD PLO EQUITY REPORT")
    print("="*60)
    
    # Player hands
    print("\nPlayer Hands:")
    for i, hand in enumerate(player_hands):
        print(f"  Player {i+1}: {' '.join(hand)}")
    
    if is_preflop:
        print("\n📊 OVERALL EQUITY (Pre-flop - identical for both boards):")
        for i, eq in enumerate(equity):
            print(f"  Player {i+1}: {eq*100:.2f}%")
    else:
        # Board-specific statistics
        print(f"\n🎯 BOARD 1 STATISTICS:")
        for i, (wins, chops) in enumerate(board1_stats):
            print(f"  Player {i+1}: Wins {wins:.2f}% | Chops {chops:.2f}%")
        
        print(f"\n🎯 BOARD 2 STATISTICS:")
        for i, (wins, chops) in enumerate(board2_stats):
            print(f"  Player {i+1}: Wins {wins:.2f}% | Chops {chops:.2f}%")
        
        print(f"\n📊 OVERALL EQUITY:")
        for i, eq in enumerate(equity):
            print(f"  Player {i+1}: {eq*100:.2f}%")
    
    print("="*60)


def generate_all_double_board_combinations(board1: List[str], board2: List[str], dealt_cards: set) -> List[Tuple[List[str], List[str]]]:
    """
    Generate all possible complete double board combinations.
    
    Args:
        board1: Cards already dealt on board1
        board2: Cards already dealt on board2
        dealt_cards: All cards already dealt (includes player hands + both boards)
    
    Returns:
        List of tuples, each containing (complete_board1, complete_board2)
    """
    # Generate all possible completions for each board independently
    complete_board1_options = generate_all_complete_boards(board1, dealt_cards)
    complete_board2_options = generate_all_complete_boards(board2, dealt_cards)
    
    board_pairs = []
    
    # Generate all combinations where board1 and board2 don't share any cards
    for b1_complete in complete_board1_options:
        for b2_complete in complete_board2_options:
            # Check if the two boards share any cards
            b1_set = set(b1_complete)
            b2_set = set(b2_complete)
            
            if b1_set.isdisjoint(b2_set):
                board_pairs.append((b1_complete, b2_complete))
    
    return board_pairs


def evaluate_single_complete_board(complete_board: List[str], all_hand_ids: List[List[int]], 
                                 evaluations: dict, board_id_map: dict, debug: bool = False) -> List[int]:
    """
    Evaluate a single complete 5-card board and return the winner indices.
    
    Args:
        complete_board: Complete 5-card board
        all_hand_ids: Hand IDs for all players
        evaluations: Pre-fetched evaluations dictionary
        board_id_map: Mapping from board keys to board IDs
        debug: Enable debug output
    
    Returns:
        List of player indices who win this board
    """
    num_players = len(all_hand_ids)
    
    # Get all 3-card subsets for this complete board
    three_card_subsets = []
    for three_card_combo in combinations(complete_board, 3):
        sorted_combo = tuple(sorted(three_card_combo, key=card_sort_key))
        three_card_subsets.append(sorted_combo)
    
    # For each player, find their best hand across all 3-card subsets
    player_best_values = []
    
    for player_idx in range(num_players):
        best_value = float('inf')  # Lower is better
        
        # Check each 3-card subset from this complete board
        for subset in three_card_subsets:
            board_key = format_board_key(subset)
            if board_key not in board_id_map:
                continue
                
            board_id = board_id_map[board_key]
            
            # Check each of this player's hand combinations with this 3-card subset
            for hand_id in all_hand_ids[player_idx]:
                if (hand_id, board_id) in evaluations:
                    value = evaluations[(hand_id, board_id)][0]  # high_hand_value
                    best_value = min(best_value, value)
        
        player_best_values.append(best_value if best_value != float('inf') else None)
    
    if debug:
        print(f"    Player best values: {player_best_values}")
    
    # Skip boards where any player has no valid evaluation
    if None in player_best_values:
        return []
    
    # Determine winners for this complete board (lowest value wins)
    best_value = min(player_best_values)
    winners = [i for i, value in enumerate(player_best_values) if value == best_value]
    
    return winners


def calculate_equity_for_multiple_hands_exhaustive(db, player_hands: List[List[str]], board: List[str] = None, debug: bool = False) -> List[float]:
    """
    Calculate equity for multiple PLO hands against all possible boards.
    
    Args:
        db: Database connection object with get_hand_ids(), get_board_ids(), and get_evaluations_for_hands_and_boards()
        player_hands: List of player hands, each hand is a list of card strings
        board: Optional list of board cards (0-5 cards)
    
    Returns:
        List of equity values (as decimals) for each player in order
    """
    if board is None:
        board = []
    
    # Get all cards in play
    dealt_cards = set()
    for hand in player_hands:
        dealt_cards.update(hand)
    dealt_cards.update(board)
    
    # Generate all possible hand combinations for each player
    player_hand_combos = generate_all_player_hand_combinations(player_hands)
    
    # Generate all possible complete 5-card boards
    complete_boards = generate_all_complete_boards(board, dealt_cards)
    
    # Get ID mappings from database
    hand_id_map = db.get_hand_ids()
    board_id_map = db.get_board_ids()
    
    # Convert hand combinations to IDs
    all_hand_ids = get_hand_ids_for_all_players(player_hand_combos, hand_id_map)
    
    # Get all unique 3-card subsets from all complete boards
    all_three_card_subsets = set()
    for complete_board in complete_boards:
        for three_card_subset in combinations(complete_board, 3):
            sorted_subset = tuple(sorted(three_card_subset, key=card_sort_key))
            all_three_card_subsets.add(sorted_subset)
    
    # Convert 3-card subsets to board IDs
    subset_board_ids = []
    for subset in all_three_card_subsets:
        board_key = format_board_key(subset)
        if board_key in board_id_map:
            subset_board_ids.append(board_id_map[board_key])
    
    # Get all unique hand IDs for database query
    unique_hand_ids = set()
    for player_hand_ids in all_hand_ids:
        unique_hand_ids.update(player_hand_ids)
    unique_hand_ids = list(unique_hand_ids)
    
    # Fetch all evaluations from database in one query
    evaluations = db.get_evaluations_for_hands_and_boards(unique_hand_ids, subset_board_ids)
    
    # Calculate equity for each complete 5-card board
    num_players = len(player_hands)
    total_equity = [0.0] * num_players
    total_boards = len(complete_boards)
    
    if debug:
        print(f"Debug: Total complete 5-card boards: {total_boards}")
        print(f"Debug: Total unique 3-card subsets: {len(all_three_card_subsets)}")
        print(f"Debug: Player hand combinations: {[len(combos) for combos in player_hand_combos]}")
    
    boards_with_ties = 0
    
    for board_idx, complete_board in enumerate(complete_boards):
        if debug and board_idx < 3:
            print(f"\nEvaluating complete board {board_idx}: {complete_board}")
        
        # Get all 3-card subsets for this complete board
        three_card_subsets = []
        for three_card_combo in combinations(complete_board, 3):
            sorted_combo = tuple(sorted(three_card_combo, key=card_sort_key))
            three_card_subsets.append(sorted_combo)
        
        # For each player, find their best hand across all 3-card subsets of this complete board
        player_best_values = []
        
        for player_idx in range(num_players):
            best_value = float('inf')  # Lower is better
            
            # Check each 3-card subset from this complete board
            for subset in three_card_subsets:
                board_key = format_board_key(subset)
                if board_key not in board_id_map:
                    continue
                    
                board_id = board_id_map[board_key]
                
                # Check each of this player's hand combinations with this 3-card subset
                for hand_id in all_hand_ids[player_idx]:
                    if (hand_id, board_id) in evaluations:
                        value = evaluations[(hand_id, board_id)][0]  # high_hand_value
                        best_value = min(best_value, value)
            
            player_best_values.append(best_value if best_value != float('inf') else None)
        
        if debug and board_idx < 3:
            print(f"  Player best values: {player_best_values}")
        
        # Skip boards where any player has no valid evaluation
        if None in player_best_values:
            if debug and board_idx < 3:
                print(f"  Skipping board due to missing evaluations")
            continue
        
        # Determine winners for this complete 5-card board (lowest value wins)
        best_value = min(player_best_values)
        winners = [i for i, value in enumerate(player_best_values) if value == best_value]
        
        if len(winners) > 1:
            boards_with_ties += 1
        
        if debug and board_idx < 3:
            print(f"  Winners: {winners} (value: {best_value})")
        
        # Distribute equity among winners
        equity_per_winner = 1.0 / len(winners)
        for winner_idx in winners:
            total_equity[winner_idx] += equity_per_winner
    
    if debug:
        print(f"\nDebug: Boards with ties: {boards_with_ties}")
        print(f"Debug: Total equity before normalization: {total_equity}")
        print(f"Debug: Final equity: {[equity / total_boards for equity in total_equity]}")
    
    # Convert to final equity percentages
    return [equity / total_boards for equity in total_equity]


def generate_all_player_hand_combinations(player_hands: List[List[str]]) -> List[List[Tuple[str, str]]]:
    """
    Generate all possible 2-card combinations for each player.
    
    Args:
        player_hands: List of player hands, each hand is a list of card strings
    
    Returns:
        List where each element is a list of 2-card tuples for that player
    """
    player_combos = []
    for hand in player_hands:
        if len(hand) < 2:
            raise ValueError(f"Hand must have at least 2 cards: {hand}")
        
        hand_combos = []
        for combo in combinations(hand, 2):
            # Sort the combo using the card sort key
            sorted_combo = tuple(sorted(combo, key=lambda x: card_sort_key(x)))
            hand_combos.append(sorted_combo)
        
        player_combos.append(hand_combos)
    
    return player_combos


def generate_all_complete_boards(current_board: List[str], dealt_cards: set) -> List[List[str]]:
    """
    Generate all possible complete 5-card boards given the current board state.
    
    Args:
        current_board: List of cards already on the board
        dealt_cards: Set of all cards already dealt (player hands + board)
    
    Returns:
        List of complete 5-card boards (as lists)
    """
    # Standard 52-card deck
    # ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    # suits = ['s', 'h', 'd', 'c']
    all_cards = [rank + suit for rank in RANKS for suit in SUITS]
    
    # Cards not yet dealt
    available_cards = [card for card in all_cards if card not in dealt_cards]
    
    complete_boards = []
    
    if len(current_board) == 0:
        # No board cards - generate all possible 5-card boards
        for five_card_board in combinations(available_cards, 5):
            complete_boards.append(list(five_card_board))
    
    elif len(current_board) == 3:
        # Flop given - generate all possible turns and rivers
        cards_needed = 2
        for additional_cards in combinations(available_cards, cards_needed):
            complete_board = current_board + list(additional_cards)
            complete_boards.append(complete_board)
    
    elif len(current_board) == 4:
        # Turn given - generate all possible rivers
        cards_needed = 1
        for additional_cards in combinations(available_cards, cards_needed):
            complete_board = current_board + list(additional_cards)
            complete_boards.append(complete_board)
    
    elif len(current_board) == 5:
        # Complete board - just return it
        complete_boards.append(current_board.copy())
    
    return complete_boards


def generate_all_possible_boards(current_board: List[str], dealt_cards: set) -> List[Tuple[str, str, str]]:
    """
    Generate all possible 3-card board combinations given the current board state.
    For incomplete boards, generates all possible runouts then extracts 3-card subsets.
    
    Args:
        current_board: List of cards already on the board
        dealt_cards: Set of all cards already dealt (player hands + board)
    
    Returns:
        List of 3-card tuples representing all possible board combinations
    """
    # Standard 52-card deck
    # ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    # suits = ['s', 'h', 'd', 'c']
    all_cards = [rank + suit for rank in RANKS for suit in SUITS]
    
    # Cards not yet dealt
    available_cards = [card for card in all_cards if card not in dealt_cards]
    
    board_combos = set()  # Use set to avoid duplicates
    
    if len(current_board) == 0:
        # No board cards - generate all possible 5-card boards, then all 3-card subsets
        for five_card_board in combinations(available_cards, 5):
            for three_card_subset in combinations(five_card_board, 3):
                sorted_combo = tuple(sorted(three_card_subset, key=card_sort_key))
                board_combos.add(sorted_combo)
    
    elif len(current_board) == 3:
        # Flop given - generate all possible turns and rivers, then all 3-card subsets
        cards_needed = 2
        for additional_cards in combinations(available_cards, cards_needed):
            complete_board = current_board + list(additional_cards)
            for three_card_subset in combinations(complete_board, 3):
                sorted_combo = tuple(sorted(three_card_subset, key=card_sort_key))
                board_combos.add(sorted_combo)
    
    elif len(current_board) == 4:
        # Turn given - generate all possible rivers, then all 3-card subsets
        cards_needed = 1
        for additional_cards in combinations(available_cards, cards_needed):
            complete_board = current_board + list(additional_cards)
            for three_card_subset in combinations(complete_board, 3):
                sorted_combo = tuple(sorted(three_card_subset, key=card_sort_key))
                board_combos.add(sorted_combo)
    
    elif len(current_board) == 5:
        # Complete board - just return all 3-card subsets
        for three_card_subset in combinations(current_board, 3):
            sorted_combo = tuple(sorted(three_card_subset, key=card_sort_key))
            board_combos.add(sorted_combo)
    
    return list(board_combos)


def get_hand_ids_for_all_players(player_hand_combos: List[List[Tuple[str, str]]], 
                                hand_id_map: Dict[str, int]) -> List[List[int]]:
    """
    Convert player hand combinations to hand IDs using the database mapping.
    
    Args:
        player_hand_combos: List of hand combinations for each player
        hand_id_map: Dictionary mapping hand strings to IDs
    
    Returns:
        List where each element is a list of hand IDs for that player
    """
    all_hand_ids = []
    
    for player_combos in player_hand_combos:
        player_hand_ids = []
        for combo in player_combos:
            hand_key = format_hand_key(combo)
            if hand_key in hand_id_map:
                player_hand_ids.append(hand_id_map[hand_key])
        all_hand_ids.append(player_hand_ids)
    
    return all_hand_ids


def format_hand_key(hand_combo: Tuple[str, str]) -> str:
    """
    Format a 2-card hand combination into the database key format.
    
    Args:
        hand_combo: Tuple of 2 card strings
    
    Returns:
        String key for database lookup
    """
    return ''.join(hand_combo)


def format_board_key(board_combo: Tuple[str, str, str]) -> str:
    """
    Format a 3-card board combination into the database key format.
    
    Args:
        board_combo: Tuple of 3 card strings
    
    Returns:
        String key for database lookup
    """
    return ''.join(board_combo)


def debug_specific_board_scenario(db, player_hands: List[List[str]], board: List[str], river_card: str):
    """
    Debug a specific river scenario to verify hand evaluation logic.
    This shows the CORRECT way to evaluate a complete 5-card board.
    """
    complete_board = board + [river_card]
    print(f"Debugging complete board: {complete_board}")
    
    # Get all 3-card combinations from this complete board
    three_card_combos = list(combinations(complete_board, 3))
    print(f"3-card board combinations: {len(three_card_combos)}")
    
    # Get hand combinations for each player
    player_hand_combos = generate_all_player_hand_combinations(player_hands)
    
    # Get ID mappings
    hand_id_map = db.get_hand_ids()
    board_id_map = db.get_board_ids()
    
    # For each player, find their BEST hand across ALL 3-card subsets
    player_best_values = []
    player_best_details = []
    
    for player_idx, hand_combos in enumerate(player_hand_combos):
        best_value = float('inf')
        best_hand = None
        best_board_subset = None
        
        print(f"\nPlayer {player_idx} evaluation:")
        
        # Check each 3-card board subset
        for three_card_combo in three_card_combos:
            sorted_combo = tuple(sorted(three_card_combo, key=card_sort_key))
            board_key = format_board_key(sorted_combo)
            
            if board_key not in board_id_map:
                continue
                
            board_id = board_id_map[board_key]
            
            # Check each of player's hand combinations with this board subset
            for hand_combo in hand_combos:
                hand_key = format_hand_key(hand_combo)
                if hand_key in hand_id_map:
                    hand_id = hand_id_map[hand_key]
                    evaluations = db.get_evaluations_for_hands_and_boards([hand_id], [board_id])
                    
                    if (hand_id, board_id) in evaluations:
                        value = evaluations[(hand_id, board_id)][0]
                        print(f"  Hand {hand_combo} + Board {sorted_combo} = {value}")
                        
                        if value < best_value:
                            best_value = value
                            best_hand = hand_combo
                            best_board_subset = sorted_combo
        
        player_best_values.append(best_value)
        player_best_details.append((best_hand, best_board_subset))
        print(f"  BEST for Player {player_idx}: {best_hand} + {best_board_subset} = {best_value}")
    
    # Now determine winner(s) for this complete 5-card board
    print(f"\n=== COMPLETE 5-CARD BOARD RESULT ===")
    print(f"Player best values: {player_best_values}")
    
    best_value = min(player_best_values)
    winners = [i for i, value in enumerate(player_best_values) if value == best_value]
    
    print(f"Winner(s): Player(s) {winners} with value {best_value}")
    return winners


def debug_complete_board_logic_error(db, player_hands: List[List[str]], board: List[str]):
    """
    Show the difference between current (incorrect) logic and correct logic.
    """
    print("=== DEMONSTRATING THE BUG ===")
    
    # Get one sample river card
    dealt_cards = set()
    for hand in player_hands:
        dealt_cards.update(hand)
    dealt_cards.update(board)
    
    # ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    # suits = ['s', 'h', 'd', 'c']
    all_cards = [rank + suit for rank in RANKS for suit in SUITS]
    available_cards = [card for card in all_cards if card not in dealt_cards]
    
    sample_river = available_cards[0]  # Just take first available card
    
    print(f"Sample complete board: {board + [sample_river]}")
    
    # Show correct logic
    print("\n--- CORRECT LOGIC (should be implemented) ---")
    winners = debug_specific_board_scenario(db, player_hands, board, sample_river)
    
    print(f"\n--- CURRENT INCORRECT LOGIC (what code actually does) ---")
    print("Current code evaluates each 3-card subset separately as independent 'boards'")
    print("This means it determines multiple winners instead of finding each player's best hand first")


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

    # 4-card Tests
    # debug_specific_board_scenario(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]], ["Tc", "7d", "7c", "2h"], "Ah")
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]])
    # print(equities)
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]], ["Tc", "7d", "7c"])
    # print(equities)
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]], ["Tc", "7d", "7c", "2h"])
    # print(equities)
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]], ["Tc", "7d", "7c", "2h", "4d"])
    # print(equities)

    # 4-card Double Board Tests
    equities = calculate_double_board_equity(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]])
    print(equities)
    equities = calculate_double_board_equity(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]], ["Tc", "7d", "7c"])
    print(equities)
    equities = calculate_double_board_equity(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]], ["Tc", "7d", "7c"], ["Ah", "Kh", "Qd"])
    print(equities)
    equities = calculate_double_board_equity(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]], ["Tc", "7d", "7c", "2h"], ["Ah", "Kh", "Qd", "4h"])
    print(equities)
    equities = calculate_double_board_equity(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]], ["Tc", "7d", "7c", "2h", "4d"], ["Ah", "Kh", "Qd", "4h", "Qc"])
    print(equities)

    # 4-card tests with 3 hands
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"], ["Ah", "Ad", "Ts", "5h"]])
    # print(equities)
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"], ["Ah", "Ad", "Ts", "5h"]], ["Tc", "7d", "7c"])
    # print(equities)
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"], ["Ah", "Ad", "Ts", "5h"]], ["Tc", "7d", "7c", "2h"])
    # print(equities)
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"], ["Ah", "Ad", "Ts", "5h"]], ["Tc", "7d", "7c", "2h", "4d"])
    # print(equities)

    # 5-card tests
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Kd", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h", "6d"]])
    # print(equities)
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Kd", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h", "6d"]], ["Tc", "7d", "7c"])
    # print(equities)
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Kd", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h", "6d"]], ["Tc", "7d", "7c", "2h"])
    # print(equities)
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Kd", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h", "6d"]], ["Tc", "7d", "7c", "2h", "4d"])
    # print(equities)

    # 6-card tests
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Kd", "Jd", "7s", "5c", "3h"], ["Qh", "Jc", "9c", "8h", "6d", "4c"]])
    # print(equities)
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Kd", "Jd", "7s", "5c", "3h"], ["Qh", "Jc", "9c", "8h", "6d", "4c"]], ["Tc", "7d", "7c"])
    # print(equities)
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Kd", "Jd", "7s", "5c", "3h"], ["Qh", "Jc", "9c", "8h", "6d", "4c"]], ["Tc", "7d", "7c", "2h"])
    # print(equities)
    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Kd", "Jd", "7s", "5c", "3h"], ["Qh", "Jc", "9c", "8h", "6d", "4c"]], ["Tc", "7d", "7c", "2h", "4d"])
    # print(equities)

   # This will show you the correct way to evaluate a single complete board
    # debug_specific_board_scenario(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]], ["Tc", "7d", "7c", "2h"], "Ah")

    # This will demonstrate the difference between correct and incorrect logic
    # debug_complete_board_logic_error(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]], ["Tc", "7d", "7c", "2h"])

    # equities = calculate_equity_for_multiple_hands_exhaustive(db, [["As", "Jd", "7s", "5c"], ["Qh", "Jc", "9c", "8h"]], ["Tc", "7d", "7c", "2h"], debug=True)
    # print(equities)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
 
    # Close the DB connection
    db.close()


if __name__ == "__main__":
    main()
