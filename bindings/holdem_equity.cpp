#include "holdem_equity.h"
#include "CardRange.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <unordered_set>
#include <functional>

namespace omp {

// Helper: parse card string to index (0-51)
static int card_string_to_index(const std::string& card_str) {
    if (card_str.length() != 2) return -1;
    
    char rank_char = card_str[0];
    char suit_char = card_str[1];
    
    // Parse rank (0-12: 2,3,4,5,6,7,8,9,T,J,Q,K,A)
    int rank;
    if (rank_char >= '2' && rank_char <= '9') {
        rank = rank_char - '2';
    } else {
        switch (rank_char) {
            case 'T': case 't': rank = 8; break;
            case 'J': case 'j': rank = 9; break;
            case 'Q': case 'q': rank = 10; break;
            case 'K': case 'k': rank = 11; break;
            case 'A': case 'a': rank = 12; break;
            default: return -1;
        }
    }
    
    // Parse suit (0-3: s,h,d,c)
    int suit;
    switch (suit_char) {
        case 's': case 'S': suit = 0; break;
        case 'h': case 'H': suit = 1; break;
        case 'd': case 'D': suit = 2; break;
        case 'c': case 'C': suit = 3; break;
        default: return -1;
    }
    
    return 4 * rank + suit;
}

// Helper: generate all k-combinations
static void generate_combinations(
    int n, int k,
    std::vector<std::vector<int>>& result)
{
    std::vector<int> combo(k);
    std::function<void(int, int)> generate = [&](int start, int pos) {
        if (pos == k) {
            result.push_back(combo);
            return;
        }
        for (int i = start; i <= n - k + pos; ++i) {
            combo[pos] = i;
            generate(i + 1, pos + 1);
        }
    };
    generate(0, 0);
}

// Evaluate best Hold'em hand: best 5 from hand + board combined
HoldemHandResult evaluate_holdem_hand(
    const std::vector<uint8_t>& hand_indices,
    const std::vector<uint8_t>& board_indices,
    HandEvaluator& eval)
{
    // Combine hand and board
    std::vector<uint8_t> all_cards;
    all_cards.reserve(hand_indices.size() + board_indices.size());
    all_cards.insert(all_cards.end(), hand_indices.begin(), hand_indices.end());
    all_cards.insert(all_cards.end(), board_indices.begin(), board_indices.end());
    
    if (all_cards.size() < 5) {
        return HoldemHandResult(0); // Invalid hand
    }
    
    uint16_t best_rank = 0;  // 0 is worst possible rank
    
    // Generate all 5-card combinations
    std::vector<std::vector<int>> combos;
    generate_combinations(all_cards.size(), 5, combos);
    
    // Evaluate all possible 5-card hands
    for (const auto& combo : combos) {
        Hand five_card = Hand::empty();
        for (int idx : combo) {
            five_card += Hand(all_cards[idx]);
        }
        
        uint16_t rank = eval.evaluate(five_card);
        if (rank > best_rank) {
            best_rank = rank;
        }
    }
    
    return HoldemHandResult(best_rank);
}

// Helper: get remaining deck
static std::vector<uint8_t> get_remaining_deck(const std::unordered_set<uint8_t>& used) {
    std::vector<uint8_t> deck;
    deck.reserve(52 - used.size());
    for (uint8_t i = 0; i < 52; ++i) {
        if (used.find(i) == used.end()) {
            deck.push_back(i);
        }
    }
    return deck;
}

// Helper: exact enumeration
static void enumerate_boards(
    const std::vector<std::vector<uint8_t>>& player_hands,
    const std::vector<uint8_t>& board_prefix,
    const std::vector<uint8_t>& remaining_deck,
    int cards_needed,
    HandEvaluator& eval,
    HoldemEquityResults& results,
    bool debug)
{
    if (cards_needed == 0) {
        // Evaluate all players
        std::vector<HoldemHandResult> player_results;
        player_results.reserve(player_hands.size());
        
        for (const auto& hand : player_hands) {
            player_results.push_back(
                evaluate_holdem_hand(hand, board_prefix, eval)
            );
        }
        
        // Find best rank
        uint16_t best_rank = 0;
        for (const auto& result : player_results) {
            if (result.high_rank > best_rank) {
                best_rank = result.high_rank;
            }
        }
        
        // Count winners
        int winner_count = 0;
        for (const auto& result : player_results) {
            if (result.high_rank == best_rank) {
                winner_count++;
            }
        }
        
        // Update statistics
        double tie_equity = 1.0 / winner_count;
        for (size_t i = 0; i < player_hands.size(); ++i) {
            if (player_results[i].high_rank == best_rank) {
                if (winner_count == 1) {
                    results.wins[i]++;
                } else {
                    results.ties[i] += tie_equity;
                }
            }
        }
        
        results.total_hands++;
        return;
    }
    
    // Recursively enumerate remaining cards
    for (size_t i = 0; i + cards_needed <= remaining_deck.size(); ++i) {
        std::vector<uint8_t> new_board = board_prefix;
        new_board.push_back(remaining_deck[i]);
        
        std::vector<uint8_t> new_deck;
        new_deck.reserve(remaining_deck.size() - 1);
        for (size_t j = i + 1; j < remaining_deck.size(); ++j) {
            new_deck.push_back(remaining_deck[j]);
        }
        
        enumerate_boards(player_hands, new_board, new_deck, cards_needed - 1, 
                        eval, results, debug);
    }
}

// Main equity computation
HoldemEquityResults compute_holdem_equity(
    const std::vector<std::vector<std::string>>& hands,
    const std::vector<std::string>& board,
    bool exact,
    uint64_t monte_carlo_samples,
    bool debug)
{
    HandEvaluator eval;
    
    if (debug) {
        std::cout << "=== Hold'em Equity Calculation ===" << std::endl;
        std::cout << "Players: " << hands.size() << std::endl;
        std::cout << "Board cards: " << board.size() << std::endl;
    }
    
    // Convert hands to indices
    std::vector<std::vector<uint8_t>> player_hands;
    std::unordered_set<uint8_t> used_cards;
    
    for (size_t p = 0; p < hands.size(); ++p) {
        std::vector<uint8_t> hand_indices;
        for (const auto& card_str : hands[p]) {
            if (card_str.empty()) continue;
            int idx = card_string_to_index(card_str);
            if (idx >= 0 && idx < 52) {
                hand_indices.push_back(static_cast<uint8_t>(idx));
                used_cards.insert(static_cast<uint8_t>(idx));
            }
        }
        player_hands.push_back(hand_indices);
    }
    
    // Convert board
    std::vector<uint8_t> board_indices;
    for (const auto& card_str : board) {
        if (card_str.empty()) continue;
        int idx = card_string_to_index(card_str);
        if (idx >= 0 && idx < 52) {
            board_indices.push_back(static_cast<uint8_t>(idx));
            used_cards.insert(static_cast<uint8_t>(idx));
        }
    }
    
    // Get remaining deck
    std::vector<uint8_t> remaining_deck = get_remaining_deck(used_cards);
    int cards_needed = 5 - board_indices.size();
    
    // Initialize results
    HoldemEquityResults results(hands.size());
    results.exact_calculation = exact;
    
    // Run calculation
    if (cards_needed == 0) {
        // Board complete - single evaluation
        std::vector<HoldemHandResult> player_results;
        for (const auto& hand : player_hands) {
            player_results.push_back(
                evaluate_holdem_hand(hand, board_indices, eval)
            );
        }
        
        uint16_t best_rank = 0;
        for (const auto& result : player_results) {
            if (result.high_rank > best_rank) {
                best_rank = result.high_rank;
            }
        }
        
        int winner_count = 0;
        for (const auto& result : player_results) {
            if (result.high_rank == best_rank) {
                winner_count++;
            }
        }
        
        double tie_equity = 1.0 / winner_count;
        for (size_t i = 0; i < hands.size(); ++i) {
            if (player_results[i].high_rank == best_rank) {
                results.equity[i] = tie_equity;
                if (winner_count == 1) {
                    results.wins[i] = 1;
                } else {
                    results.ties[i] = tie_equity;
                }
            }
        }
        results.total_hands = 1;
        
    } else {
        // Exact enumeration
        enumerate_boards(player_hands, board_indices, remaining_deck, 
                        cards_needed, eval, results, debug);
        
        for (size_t i = 0; i < hands.size(); ++i) {
            results.equity[i] = (results.wins[i] + results.ties[i]) / results.total_hands;
        }
    }
    
    if (debug) {
        std::cout << "\n=== Results ===" << std::endl;
        for (size_t i = 0; i < hands.size(); ++i) {
            std::cout << "Player " << (i+1) << ": " 
                     << (results.equity[i] * 100) << "% " << std::endl;
        }
    }
    
    return results;
}

} // namespace omp