#include "omaha_equity.h"
#include "CardRange.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <unordered_set>
#include <functional>
#include <cstring>

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

// Helper: generate all k-combinations of indices from 0 to n-1
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

// Evaluate best Omaha hand: exactly 2 from hand, 3 from board
OmahaHandResult evaluate_omaha_hand(
    const std::vector<uint8_t>& hand_indices,
    const std::vector<uint8_t>& board_indices,
    HandEvaluator& eval)
{
    if (hand_indices.size() < 2 || board_indices.size() != 5) {
        return OmahaHandResult(0); // Invalid hand
    }
    
    uint16_t best_rank = 0;  // 0 is worst possible rank
    
    // Generate all 2-card combinations from hand
    std::vector<std::vector<int>> hand_combos;
    generate_combinations(hand_indices.size(), 2, hand_combos);
    
    // Generate all 3-card combinations from board (always 10 combos for 5 cards)
    std::vector<std::vector<int>> board_combos;
    generate_combinations(5, 3, board_combos);
    
    // Evaluate all possible 5-card hands
    for (const auto& hand_combo : hand_combos) {
        for (const auto& board_combo : board_combos) {
            // Build 5-card hand
            Hand five_card = Hand::empty();
            five_card += Hand(hand_indices[hand_combo[0]]);
            five_card += Hand(hand_indices[hand_combo[1]]);
            five_card += Hand(board_indices[board_combo[0]]);
            five_card += Hand(board_indices[board_combo[1]]);
            five_card += Hand(board_indices[board_combo[2]]);
            
            uint16_t rank = eval.evaluate(five_card);
            if (rank > best_rank) {
                best_rank = rank;
            }
        }
    }
    
    return OmahaHandResult(best_rank);
}

// Helper: get all remaining cards not in used set
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

// Helper: exact enumeration of all board runouts
static void enumerate_boards(
    const std::vector<std::vector<uint8_t>>& player_hands,
    const std::vector<uint8_t>& board_prefix,
    const std::vector<uint8_t>& remaining_deck,
    int cards_needed,
    HandEvaluator& eval,
    OmahaEquityResults& results,
    bool debug)
{
    if (cards_needed == 0) {
        // Evaluate all players with complete board
        std::vector<OmahaHandResult> player_results;
        player_results.reserve(player_hands.size());
        
        for (const auto& hand : player_hands) {
            player_results.push_back(
                evaluate_omaha_hand(hand, board_prefix, eval)
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
    // We need to iterate through remaining deck, picking cards in order to avoid duplicates
    for (size_t i = 0; i + cards_needed <= remaining_deck.size(); ++i) {
        std::vector<uint8_t> new_board = board_prefix;
        new_board.push_back(remaining_deck[i]);
        
        // Create new deck without the card we just added
        std::vector<uint8_t> new_deck;
        new_deck.reserve(remaining_deck.size() - 1);
        for (size_t j = i + 1; j < remaining_deck.size(); ++j) {
            new_deck.push_back(remaining_deck[j]);
        }
        
        enumerate_boards(player_hands, new_board, new_deck, cards_needed - 1, 
                        eval, results, debug);
    }
}

// Helper: Monte Carlo sampling of board runouts
static void monte_carlo_boards(
    const std::vector<std::vector<uint8_t>>& player_hands,
    const std::vector<uint8_t>& board_prefix,
    const std::vector<uint8_t>& remaining_deck,
    int cards_needed,
    uint64_t num_samples,
    HandEvaluator& eval,
    OmahaEquityResults& results,
    bool debug)
{
    std::mt19937_64 rng(std::random_device{}());
    
    for (uint64_t sample = 0; sample < num_samples; ++sample) {
        // Shuffle deck and take first cards_needed cards
        std::vector<uint8_t> deck_copy = remaining_deck;
        std::shuffle(deck_copy.begin(), deck_copy.end(), rng);
        
        std::vector<uint8_t> complete_board = board_prefix;
        for (int i = 0; i < cards_needed; ++i) {
            complete_board.push_back(deck_copy[i]);
        }
        
        // Evaluate all players
        std::vector<OmahaHandResult> player_results;
        player_results.reserve(player_hands.size());
        
        for (const auto& hand : player_hands) {
            player_results.push_back(
                evaluate_omaha_hand(hand, complete_board, eval)
            );
        }
        
        // Find winner(s)
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
    }
}

// Main equity computation function
OmahaEquityResults compute_omaha_equity(
    const std::vector<std::vector<std::string>>& hands,
    const std::vector<std::string>& board,
    bool exact,
    uint64_t monte_carlo_samples,
    bool debug)
{
    HandEvaluator eval;
    
    if (debug) {
        std::cout << "=== Omaha Equity Calculation ===" << std::endl;
        std::cout << "Players: " << hands.size() << std::endl;
        std::cout << "Board cards: " << board.size() << std::endl;
        std::cout << "Mode: " << (exact ? "Exact" : "Monte Carlo") << std::endl;
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
            } else if (debug) {
                std::cout << "Warning: invalid card '" << card_str << "' for player " << (p+1) << std::endl;
            }
        }
        
        if (hand_indices.size() < 2) {
            if (debug) {
                std::cout << "Warning: player " << (p+1) << " has fewer than 2 valid cards" << std::endl;
            }
        }
        player_hands.push_back(hand_indices);
    }
    
    // Convert board to indices
    std::vector<uint8_t> board_indices;
    for (const auto& card_str : board) {
        if (card_str.empty()) continue;
        int idx = card_string_to_index(card_str);
        if (idx >= 0 && idx < 52) {
            board_indices.push_back(static_cast<uint8_t>(idx));
            used_cards.insert(static_cast<uint8_t>(idx));
        } else if (debug) {
            std::cout << "Warning: invalid board card '" << card_str << "'" << std::endl;
        }
    }
    
    if (board_indices.size() > 5) {
        if (debug) {
            std::cout << "Error: board has more than 5 cards" << std::endl;
        }
        return OmahaEquityResults(hands.size());
    }
    
    // Get remaining deck
    std::vector<uint8_t> remaining_deck = get_remaining_deck(used_cards);
    int cards_needed = 5 - board_indices.size();
    
    if (debug) {
        std::cout << "Cards needed for complete board: " << cards_needed << std::endl;
        std::cout << "Remaining deck size: " << remaining_deck.size() << std::endl;
    }
    
    // Initialize results
    OmahaEquityResults results(hands.size());
    results.exact_calculation = exact;
    
    // Run calculation
    if (cards_needed == 0) {
        // Board is complete - single evaluation
        std::vector<OmahaHandResult> player_results;
        for (const auto& hand : player_hands) {
            player_results.push_back(
                evaluate_omaha_hand(hand, board_indices, eval)
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
        
    } else if (exact) {
        // Exact enumeration
        enumerate_boards(player_hands, board_indices, remaining_deck, 
                        cards_needed, eval, results, debug);
        
        // Calculate equity percentages
        for (size_t i = 0; i < hands.size(); ++i) {
            results.equity[i] = (results.wins[i] + results.ties[i]) / results.total_hands;
        }
        
    } else {
        // Monte Carlo sampling
        monte_carlo_boards(player_hands, board_indices, remaining_deck,
                          cards_needed, monte_carlo_samples, eval, results, debug);
        
        // Calculate equity percentages
        for (size_t i = 0; i < hands.size(); ++i) {
            results.equity[i] = (results.wins[i] + results.ties[i]) / results.total_hands;
        }
    }
    
    if (debug) {
        std::cout << "\n=== Results ===" << std::endl;
        for (size_t i = 0; i < hands.size(); ++i) {
            std::cout << "Player " << (i+1) << ": " 
                     << (results.equity[i] * 100) << "% "
                     << "(wins: " << results.wins[i] 
                     << ", ties: " << results.ties[i] << ")" << std::endl;
        }
        std::cout << "Total hands evaluated: " << results.total_hands << std::endl;
    }
    
    return results;
}

} // namespace omp

// #include "omaha_equity.h"
// #include "CardRange.h"
// #include <algorithm>
// #include <random>
// #include <iostream>
// #include <unordered_set>
// #include <functional>
// #include <cstring>

// namespace omp {

// // Helper: parse card string to index (0-51)
// static int card_string_to_index(const std::string& card_str) {
//     if (card_str.length() != 2) return -1;
    
//     char rank_char = card_str[0];
//     char suit_char = card_str[1];
    
//     // Parse rank (0-12: 2,3,4,5,6,7,8,9,T,J,Q,K,A)
//     int rank;
//     if (rank_char >= '2' && rank_char <= '9') {
//         rank = rank_char - '2';
//     } else {
//         switch (rank_char) {
//             case 'T': case 't': rank = 8; break;
//             case 'J': case 'j': rank = 9; break;
//             case 'Q': case 'q': rank = 10; break;
//             case 'K': case 'k': rank = 11; break;
//             case 'A': case 'a': rank = 12; break;
//             default: return -1;
//         }
//     }
    
//     // Parse suit (0-3: s,h,d,c)
//     int suit;
//     switch (suit_char) {
//         case 's': case 'S': suit = 0; break;
//         case 'h': case 'H': suit = 1; break;
//         case 'd': case 'D': suit = 2; break;
//         case 'c': case 'C': suit = 3; break;
//         default: return -1;
//     }
    
//     return 4 * rank + suit;
// }

// // Helper: generate all k-combinations of indices from 0 to n-1
// static void generate_combinations(
//     int n, int k,
//     std::vector<std::vector<int>>& result)
// {
//     std::vector<int> combo(k);
//     std::function<void(int, int)> generate = [&](int start, int pos) {
//         if (pos == k) {
//             result.push_back(combo);
//             return;
//         }
//         for (int i = start; i <= n - k + pos; ++i) {
//             combo[pos] = i;
//             generate(i + 1, pos + 1);
//         }
//     };
//     generate(0, 0);
// }

// // Evaluate best Omaha hand: exactly 2 from hand, 3 from board
// OmahaHandResult evaluate_omaha_hand(
//     const std::vector<uint8_t>& hand_indices,
//     const std::vector<uint8_t>& board_indices,
//     HandEvaluator& eval)
// {
//     if (hand_indices.size() < 2 || board_indices.size() != 5) {
//         return OmahaHandResult(0); // Invalid hand
//     }
    
//     uint16_t best_rank = 0;  // 0 is worst possible rank
    
//     // Generate all 2-card combinations from hand
//     std::vector<std::vector<int>> hand_combos;
//     generate_combinations(hand_indices.size(), 2, hand_combos);
    
//     // Generate all 3-card combinations from board (always 10 combos for 5 cards)
//     std::vector<std::vector<int>> board_combos;
//     generate_combinations(5, 3, board_combos);
    
//     // Evaluate all possible 5-card hands
//     for (const auto& hand_combo : hand_combos) {
//         for (const auto& board_combo : board_combos) {
//             // Build 5-card hand
//             Hand five_card = Hand::empty();
//             five_card += Hand(hand_indices[hand_combo[0]]);
//             five_card += Hand(hand_indices[hand_combo[1]]);
//             five_card += Hand(board_indices[board_combo[0]]);
//             five_card += Hand(board_indices[board_combo[1]]);
//             five_card += Hand(board_indices[board_combo[2]]);
            
//             uint16_t rank = eval.evaluate(five_card);
//             if (rank > best_rank) {
//                 best_rank = rank;
//             }
//         }
//     }
    
//     return OmahaHandResult(best_rank);
// }

// // Helper: get all remaining cards not in used set
// static std::vector<uint8_t> get_remaining_deck(const std::unordered_set<uint8_t>& used) {
//     std::vector<uint8_t> deck;
//     deck.reserve(52 - used.size());
//     for (uint8_t i = 0; i < 52; ++i) {
//         if (used.find(i) == used.end()) {
//             deck.push_back(i);
//         }
//     }
//     return deck;
// }

// // Helper: exact enumeration of all board runouts
// static void enumerate_boards(
//     const std::vector<std::vector<uint8_t>>& player_hands,
//     const std::vector<uint8_t>& board_prefix,
//     const std::vector<uint8_t>& remaining_deck,
//     int cards_needed,
//     HandEvaluator& eval,
//     OmahaEquityResults& results,
//     bool debug)
// {
//     if (cards_needed == 0) {
//         // Evaluate all players with complete board
//         std::vector<OmahaHandResult> player_results;
//         player_results.reserve(player_hands.size());
        
//         for (const auto& hand : player_hands) {
//             player_results.push_back(
//                 evaluate_omaha_hand(hand, board_prefix, eval)
//             );
//         }
        
//         // Find best rank
//         uint16_t best_rank = 0;
//         for (const auto& result : player_results) {
//             if (result.high_rank > best_rank) {
//                 best_rank = result.high_rank;
//             }
//         }
        
//         // Count winners
//         int winner_count = 0;
//         for (const auto& result : player_results) {
//             if (result.high_rank == best_rank) {
//                 winner_count++;
//             }
//         }
        
//         // Update statistics
//         double tie_equity = 1.0 / winner_count;
//         for (size_t i = 0; i < player_hands.size(); ++i) {
//             if (player_results[i].high_rank == best_rank) {
//                 if (winner_count == 1) {
//                     results.wins[i]++;
//                 } else {
//                     results.ties[i] += tie_equity;
//                 }
//             }
//         }
        
//         results.total_hands++;
//         return;
//     }
    
//     // Recursively enumerate remaining cards
//     int start_idx = board_prefix.empty() ? 0 : 
//         std::find(remaining_deck.begin(), remaining_deck.end(), board_prefix.back()) - remaining_deck.begin() + 1;
    
//     for (size_t i = start_idx; i <= remaining_deck.size() - cards_needed; ++i) {
//         std::vector<uint8_t> new_board = board_prefix;
//         new_board.push_back(remaining_deck[i]);
        
//         std::vector<uint8_t> new_deck;
//         new_deck.reserve(remaining_deck.size() - 1);
//         for (size_t j = 0; j < remaining_deck.size(); ++j) {
//             if (j != i) {
//                 new_deck.push_back(remaining_deck[j]);
//             }
//         }
        
//         enumerate_boards(player_hands, new_board, new_deck, cards_needed - 1, 
//                         eval, results, debug);
//     }
// }

// // Helper: Monte Carlo sampling of board runouts
// static void monte_carlo_boards(
//     const std::vector<std::vector<uint8_t>>& player_hands,
//     const std::vector<uint8_t>& board_prefix,
//     const std::vector<uint8_t>& remaining_deck,
//     int cards_needed,
//     uint64_t num_samples,
//     HandEvaluator& eval,
//     OmahaEquityResults& results,
//     bool debug)
// {
//     std::mt19937_64 rng(std::random_device{}());
    
//     for (uint64_t sample = 0; sample < num_samples; ++sample) {
//         // Shuffle deck and take first cards_needed cards
//         std::vector<uint8_t> deck_copy = remaining_deck;
//         std::shuffle(deck_copy.begin(), deck_copy.end(), rng);
        
//         std::vector<uint8_t> complete_board = board_prefix;
//         for (int i = 0; i < cards_needed; ++i) {
//             complete_board.push_back(deck_copy[i]);
//         }
        
//         // Evaluate all players
//         std::vector<OmahaHandResult> player_results;
//         player_results.reserve(player_hands.size());
        
//         for (const auto& hand : player_hands) {
//             player_results.push_back(
//                 evaluate_omaha_hand(hand, complete_board, eval)
//             );
//         }
        
//         // Find winner(s)
//         uint16_t best_rank = 0;
//         for (const auto& result : player_results) {
//             if (result.high_rank > best_rank) {
//                 best_rank = result.high_rank;
//             }
//         }
        
//         int winner_count = 0;
//         for (const auto& result : player_results) {
//             if (result.high_rank == best_rank) {
//                 winner_count++;
//             }
//         }
        
//         // Update statistics
//         double tie_equity = 1.0 / winner_count;
//         for (size_t i = 0; i < player_hands.size(); ++i) {
//             if (player_results[i].high_rank == best_rank) {
//                 if (winner_count == 1) {
//                     results.wins[i]++;
//                 } else {
//                     results.ties[i] += tie_equity;
//                 }
//             }
//         }
        
//         results.total_hands++;
//     }
// }

// // Main equity computation function
// OmahaEquityResults compute_omaha_equity(
//     const std::vector<std::vector<std::string>>& hands,
//     const std::vector<std::string>& board,
//     bool exact,
//     uint64_t monte_carlo_samples,
//     bool debug)
// {
//     HandEvaluator eval;
    
//     if (debug) {
//         std::cout << "=== Omaha Equity Calculation ===" << std::endl;
//         std::cout << "Players: " << hands.size() << std::endl;
//         std::cout << "Board cards: " << board.size() << std::endl;
//         std::cout << "Mode: " << (exact ? "Exact" : "Monte Carlo") << std::endl;
//     }
    
//     // Convert hands to indices
//     std::vector<std::vector<uint8_t>> player_hands;
//     std::unordered_set<uint8_t> used_cards;
    
//     for (size_t p = 0; p < hands.size(); ++p) {
//         std::vector<uint8_t> hand_indices;
//         for (const auto& card_str : hands[p]) {
//             if (card_str.empty()) continue;
//             int idx = card_string_to_index(card_str);
//             if (idx >= 0 && idx < 52) {
//                 hand_indices.push_back(static_cast<uint8_t>(idx));
//                 used_cards.insert(static_cast<uint8_t>(idx));
//             } else if (debug) {
//                 std::cout << "Warning: invalid card '" << card_str << "' for player " << (p+1) << std::endl;
//             }
//         }
        
//         if (hand_indices.size() < 2) {
//             if (debug) {
//                 std::cout << "Warning: player " << (p+1) << " has fewer than 2 valid cards" << std::endl;
//             }
//         }
//         player_hands.push_back(hand_indices);
//     }
    
//     // Convert board to indices
//     std::vector<uint8_t> board_indices;
//     for (const auto& card_str : board) {
//         if (card_str.empty()) continue;
//         int idx = card_string_to_index(card_str);
//         if (idx >= 0 && idx < 52) {
//             board_indices.push_back(static_cast<uint8_t>(idx));
//             used_cards.insert(static_cast<uint8_t>(idx));
//         } else if (debug) {
//             std::cout << "Warning: invalid board card '" << card_str << "'" << std::endl;
//         }
//     }
    
//     if (board_indices.size() > 5) {
//         if (debug) {
//             std::cout << "Error: board has more than 5 cards" << std::endl;
//         }
//         return OmahaEquityResults(hands.size());
//     }
    
//     // Get remaining deck
//     std::vector<uint8_t> remaining_deck = get_remaining_deck(used_cards);
//     int cards_needed = 5 - board_indices.size();
    
//     if (debug) {
//         std::cout << "Cards needed for complete board: " << cards_needed << std::endl;
//         std::cout << "Remaining deck size: " << remaining_deck.size() << std::endl;
//     }
    
//     // Initialize results
//     OmahaEquityResults results(hands.size());
//     results.exact_calculation = exact;
    
//     // Run calculation
//     if (cards_needed == 0) {
//         // Board is complete - single evaluation
//         std::vector<OmahaHandResult> player_results;
//         for (const auto& hand : player_hands) {
//             player_results.push_back(
//                 evaluate_omaha_hand(hand, board_indices, eval)
//             );
//         }
        
//         uint16_t best_rank = 0;
//         for (const auto& result : player_results) {
//             if (result.high_rank > best_rank) {
//                 best_rank = result.high_rank;
//             }
//         }
        
//         int winner_count = 0;
//         for (const auto& result : player_results) {
//             if (result.high_rank == best_rank) {
//                 winner_count++;
//             }
//         }
        
//         double tie_equity = 1.0 / winner_count;
//         for (size_t i = 0; i < hands.size(); ++i) {
//             if (player_results[i].high_rank == best_rank) {
//                 results.equity[i] = tie_equity;
//                 if (winner_count == 1) {
//                     results.wins[i] = 1;
//                 } else {
//                     results.ties[i] = tie_equity;
//                 }
//             }
//         }
//         results.total_hands = 1;
        
//     } else if (exact) {
//         // Exact enumeration
//         enumerate_boards(player_hands, board_indices, remaining_deck, 
//                         cards_needed, eval, results, debug);
        
//         // Calculate equity percentages
//         for (size_t i = 0; i < hands.size(); ++i) {
//             results.equity[i] = (results.wins[i] + results.ties[i]) / results.total_hands;
//         }
        
//     } else {
//         // Monte Carlo sampling
//         monte_carlo_boards(player_hands, board_indices, remaining_deck,
//                           cards_needed, monte_carlo_samples, eval, results, debug);
        
//         // Calculate equity percentages
//         for (size_t i = 0; i < hands.size(); ++i) {
//             results.equity[i] = (results.wins[i] + results.ties[i]) / results.total_hands;
//         }
//     }
    
//     if (debug) {
//         std::cout << "\n=== Results ===" << std::endl;
//         for (size_t i = 0; i < hands.size(); ++i) {
//             std::cout << "Player " << (i+1) << ": " 
//                      << (results.equity[i] * 100) << "% "
//                      << "(wins: " << results.wins[i] 
//                      << ", ties: " << results.ties[i] << ")" << std::endl;
//         }
//         std::cout << "Total hands evaluated: " << results.total_hands << std::endl;
//     }
    
//     return results;
// }

// } // namespace omp