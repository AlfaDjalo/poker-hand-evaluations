#ifndef HOLDEM_EQUITY_H
#define HOLDEM_EQUITY_H

#include <vector>
#include <string>
#include "HandEvaluator.h"

namespace omp {

// Result structure for Hold'em hand evaluation
struct HoldemHandResult {
    uint16_t high_rank;
    
    HoldemHandResult(uint16_t rank = 0) : high_rank(rank) {}
};

// Result structure for equity calculations
struct HoldemEquityResults {
    std::vector<double> equity;
    std::vector<double> wins;
    std::vector<double> ties;
    uint64_t total_hands;
    bool exact_calculation;
    
    HoldemEquityResults(size_t num_players)
        : equity(num_players, 0.0),
          wins(num_players, 0.0),
          ties(num_players, 0.0),
          total_hands(0),
          exact_calculation(true) {}
};

// Evaluate best Hold'em hand: best 5 cards from hand + board
HoldemHandResult evaluate_holdem_hand(
    const std::vector<uint8_t>& hand_indices,
    const std::vector<uint8_t>& board_indices,
    HandEvaluator& eval);

// Compute Hold'em equity for given hands and board
HoldemEquityResults compute_holdem_equity(
    const std::vector<std::vector<std::string>>& hands,
    const std::vector<std::string>& board = std::vector<std::string>(),
    bool exact = true,
    uint64_t monte_carlo_samples = 100000,
    bool debug = false);

} // namespace omp

#endif // HOLDEM_EQUITY_H