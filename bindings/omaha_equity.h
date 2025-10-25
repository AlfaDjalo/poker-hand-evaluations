#ifndef OMAHA_EQUITY_H
#define OMAHA_EQUITY_H

#include "HandEvaluator.h"
#include <vector>
#include <string>
#include <cstdint>

namespace omp {

// Result for a single hand evaluation (extensible to hi/lo or multi-board)
struct OmahaHandResult {
    uint16_t high_rank;  // Hand rank for high (lower value = better hand)
    
    // Reserved for future extensions:
    // uint16_t low_rank;   // For Omaha Hi/Lo
    // uint16_t board_ranks[3];  // For multi-board games
    
    OmahaHandResult() : high_rank(0) {}
    explicit OmahaHandResult(uint16_t high) : high_rank(high) {}
    
    // Comparison for determining winner (extensible)
    bool beats(const OmahaHandResult& other) const {
        return high_rank > other.high_rank;  // Higher rank value is better
    }
    
    bool ties(const OmahaHandResult& other) const {
        return high_rank == other.high_rank;
    }
};

// Equity calculation results
struct OmahaEquityResults {
    std::vector<double> equity;     // Equity percentage for each player (0-1)
    std::vector<uint64_t> wins;     // Win count for each player
    std::vector<double> ties;       // Tie count adjusted for split size
    uint64_t total_hands;           // Total hands evaluated
    bool exact_calculation;         // True if exact enumeration, false if Monte Carlo
    
    OmahaEquityResults(size_t num_players) 
        : equity(num_players, 0.0),
          wins(num_players, 0),
          ties(num_players, 0.0),
          total_hands(0),
          exact_calculation(true) {}
};

// Evaluate the best Omaha hand using exactly 2 cards from hand and 3 from board
// hand_indices: card indices (0-51), must have at least 2 cards
// board_indices: exactly 5 card indices (0-51)
// Returns OmahaHandResult with hand rank (extensible to hi/lo)
OmahaHandResult evaluate_omaha_hand(
    const std::vector<uint8_t>& hand_indices,
    const std::vector<uint8_t>& board_indices,
    HandEvaluator& eval
);

// Compute Omaha equity for multiple players
// hands: vector of hand card strings for each player (e.g. [["As","Kd","Qh","Jc"], ...])
// board: board card strings (0-5 cards)
// exact: true for exact enumeration, false for Monte Carlo
// monte_carlo_samples: number of samples for Monte Carlo (ignored if exact=true)
// debug: print debug information
OmahaEquityResults compute_omaha_equity(
    const std::vector<std::vector<std::string>>& hands,
    const std::vector<std::string>& board,
    bool exact = true,
    uint64_t monte_carlo_samples = 100000,
    bool debug = false
);

} // namespace omp

#endif // OMAHA_EQUITY_H