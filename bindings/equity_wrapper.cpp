// equity_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "omaha_equity.h"
#include <vector>
#include <string>

namespace py = pybind11;
using namespace omp;

// Wrapper function that returns a dict with detailed results
py::dict compute_equity(
    const std::vector<std::vector<std::string>>& hands,
    const std::vector<std::string>& board = std::vector<std::string>(),
    bool exact = true,
    uint64_t monte_carlo_samples = 100000,
    bool debug = false)
{
    OmahaEquityResults results = compute_omaha_equity(
        hands, board, exact, monte_carlo_samples, debug
    );
    
    py::dict output;
    output["equities"] = results.equity;
    output["wins"] = results.wins;
    output["ties"] = results.ties;
    output["total_hands"] = results.total_hands;
    output["exact"] = results.exact_calculation;
    
    return output;
}

// Python bindings
PYBIND11_MODULE(equity_wrapper, m) {
    m.doc() = "Omaha equity calculator using OMPEval";

    m.def("compute_equity", &compute_equity,
          py::arg("hands"),
          py::arg("board") = std::vector<std::string>(),
          py::arg("exact") = true,
          py::arg("monte_carlo_samples") = 100000,
          py::arg("debug") = false,
          "Compute Omaha equities for given hands and board.\n\n"
          "Args:\n"
          "  hands: List of player hands, each hand is a list of card strings\n"
          "         (e.g. [['As','Kd','Qh','Jc'], ['2h','3h','4h','5h']])\n"
          "  board: List of board card strings (0-5 cards)\n"
          "  exact: If True, use exact enumeration; if False, use Monte Carlo\n"
          "  monte_carlo_samples: Number of samples for Monte Carlo (ignored if exact=True)\n"
          "  debug: Print debug info to stdout\n\n"
          "Returns:\n"
          "  Dictionary with keys:\n"
          "    'equities': List of equity values (0-1) for each player\n"
          "    'wins': List of win counts for each player\n"
          "    'ties': List of tie counts (adjusted for split size)\n"
          "    'total_hands': Total number of hands evaluated\n"
          "    'exact': Boolean indicating if exact calculation was used");
}




// // equity_wrapper.cpp
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

// #include <vector>
// #include <string>
// #include <array>
// #include <cstdint>
// #include <iostream>

// #include "EquityCalculator.h"
// #include "CardRange.h"

// namespace py = pybind11;
// using namespace omp;

// // Helper: from a single-card string (like "As" or "Td") produce the single-card bitmask using OMPEval's parser.
// static uint64_t single_card_mask(const std::string &cardStr)
// {
//     // CardRange::getCardMask accepts a string like "As" or "2c" or multiple cards "AsKd"
//     // We feed it a single card and expect a one-bit mask if valid.
//     return CardRange::getCardMask(cardStr);
// }

// // Helper: get numeric card index 0..63 (really 0..51) from a single-bit mask. Returns -1 if mask==0 or not single-bit.
// static int card_index_from_mask(uint64_t mask)
// {
//     if (mask == 0) return -1;
//     // find single set bit index (0..63)
//     // mask should have only one bit set for a single card; if not, still pick lowest set bit.
//     for (int i = 0; i < 64; ++i) {
//         if ((mask >> i) & 1ull) return i;
//     }
//     return -1;
// }

// std::vector<double> compute_equity(
//     const std::vector<std::vector<std::string>>& hands_in,
//     const std::vector<std::string>& board_in,
//     bool debug = false)
// {
//     if (debug) std::cout << "=== compute_equity called ===\n";

//     // Filter and normalize incoming hands: remove empty strings / None equivalents already filtered in Python,
//     // but defensively handle them here.
//     std::vector<std::vector<std::string>> hands;
//     for (size_t p = 0; p < hands_in.size(); ++p) {
//         std::vector<std::string> filtered;
//         for (const auto &c : hands_in[p]) {
//             if (!c.empty()) filtered.push_back(c);
//         }
//         hands.push_back(std::move(filtered));
//     }

//     // Filter board cards
//     std::vector<std::string> board;
//     for (const auto &c : board_in) {
//         if (!c.empty()) board.push_back(c);
//     }

//     if (debug) {
//         std::cout << "Filtered player hands: [\n";
//         for (size_t i = 0; i < hands.size(); ++i) {
//             std::cout << "  Player " << (i+1) << ":";
//             for (auto &c : hands[i]) std::cout << " " << c;
//             std::cout << "\n";
//         }
//         std::cout << "]\n";
//         std::cout << "Filtered board: [";
//         for (auto &c : board) std::cout << c << " ";
//         std::cout << "]\n";
//     }

//     // Quick validations
//     const size_t nPlayers = hands.size();
//     if (nPlayers == 0) {
//         if (debug) std::cout << "No players => returning empty vector\n";
//         return {};
//     }

//     // Build per-card mask table for board and for every card string encountered
//     // Also build numeric indices for cards (0..51) by extracting the bit position from the one-bit mask.
//     // We'll need numeric card indices when building CardRange via vector-of-combos constructor.
//     // Map each distinct card string encountered to mask and index.
//     std::unordered_map<std::string, uint64_t> cardMaskMap;
//     std::unordered_map<std::string, int> cardIndexMap;

//     auto ensure_card = [&](const std::string &cstr) -> bool {
//         if (cstr.empty()) return false;
//         if (cardMaskMap.find(cstr) != cardMaskMap.end()) return true;
//         uint64_t mask = single_card_mask(cstr);
//         if (mask == 0) {
//             if (debug) std::cout << "Warning: unable to parse card string \"" << cstr << "\" -> mask==0\n";
//             return false;
//         }
//         int idx = card_index_from_mask(mask);
//         if (idx < 0) {
//             if (debug) std::cout << "Warning: could not derive index from mask for \"" << cstr << "\"\n";
//             return false;
//         }
//         cardMaskMap[cstr] = mask;
//         cardIndexMap[cstr] = idx;
//         return true;
//     };

//     // Ensure all player cards and board cards are known
//     for (const auto &ph : hands) {
//         for (const auto &c : ph) {
//             ensure_card(c);
//         }
//     }
//     for (const auto &c : board) {
//         ensure_card(c);
//     }

//     // If any player's valid card count < 2, we should still allow calculation but skip that player (or return zeros).
//     // For now we will detect players with <2 valid cards and warn + treat as empty (equity 0).
//     std::vector<bool> playerHasEnoughCards(nPlayers, true);
//     for (size_t p = 0; p < nPlayers; ++p) {
//         size_t valid = 0;
//         for (auto &c : hands[p]) {
//             if (cardMaskMap.find(c) != cardMaskMap.end()) ++valid;
//         }
//         if (valid < 2) {
//             playerHasEnoughCards[p] = false;
//             if (debug)
//                 std::cout << "Warning: player " << (p+1) << " has fewer than 2 valid cards -> they will be ignored\n";
//         }
//     }

//     // Convert board into a single bitmask (uint64_t)
//     uint64_t boardMask = 0ull;
//     for (auto &c : board) {
//         auto it = cardMaskMap.find(c);
//         if (it != cardMaskMap.end()) boardMask |= it->second;
//     }
//     if (debug) {
//         std::cout << "Board mask = 0x" << std::hex << boardMask << std::dec << "\n";
//     }

//     // Build CardRange objects for each player from their explicit 2-card combos obtained from the given hole cards.
//     // For each player, enumerate all unordered pairs (i<j) of their valid cards and add that pair as a combo
//     // to the vector<array<uint8_t,2>> constructor for CardRange.
//     std::vector<CardRange> cardRanges;
//     cardRanges.reserve(nPlayers);

//     for (size_t p = 0; p < nPlayers; ++p) {
//         std::vector<std::array<uint8_t,2>> combos;
//         // Collect the valid card indices for this player in a temporary vector
//         std::vector<int> indices;
//         for (const auto &c : hands[p]) {
//             auto it = cardIndexMap.find(c);
//             if (it != cardIndexMap.end()) indices.push_back(it->second);
//             else {
//                 if (debug) std::cout << "Skipping unknown card '" << c << "' for player " << (p+1) << "\n";
//             }
//         }

//         // enumerate unordered pairs (i,j) i<j
//         for (size_t i = 0; i < indices.size(); ++i) {
//             for (size_t j = i+1; j < indices.size(); ++j) {
//                 std::array<uint8_t,2> combo = { static_cast<uint8_t>(indices[i]),
//                                                 static_cast<uint8_t>(indices[j]) };
//                 combos.push_back(combo);
//             }
//         }

//         if (combos.empty()) {
//             // create an empty CardRange (constructor default) so OMPEval knows player has no combos
//             CardRange cr;
//             cardRanges.push_back(std::move(cr));
//             if (debug) std::cout << "Player " << (p+1) << " -> 0 combos\n";
//         } else {
//             CardRange cr(combos);
//             cardRanges.push_back(std::move(cr));
//             if (debug) std::cout << "Player " << (p+1) << " -> " << combos.size() << " combos\n";
//         }
//     }

//     // Create EquityCalculator and run exact enumeration
//     EquityCalculator calc;
//     // enumerateAll true -> exact enumeration; small stdevTarget ignored for enumeration.
//     bool enumerateAll = true;
//     double stdevTarget = 0.0; // ignored when enumerateAll == true

//     if (debug) std::cout << "Starting calculation...\n";
//     bool started = calc.start(cardRanges, boardMask, 0ULL, enumerateAll, stdevTarget, nullptr, 0.1, 0);
//     if (!started) {
//         if (debug) std::cout << "calc.start() returned false -> impossible combination or other error\n";
//         // return a zero-filled vector for each player
//         return std::vector<double>(nPlayers, 0.0);
//     }
//     calc.wait();

//     auto res = calc.getResults();

//     if (debug) {
//         std::cout << "Calculation complete.\n";
//         std::cout << "Players: " << res.players << "\n";
//         // Show wins/hands optionally
//         std::cout << "Hands processed: " << res.hands << "\n";
//     }

//     // res.equity is an array sized MAX_PLAYERS. Only return the first res.players entries.
//     unsigned actualPlayers = res.players;
//     if (actualPlayers == 0) {
//         // fallback: if res.players was not populated, use the input count (but may be wrong)
//         actualPlayers = static_cast<unsigned>(nPlayers);
//     }

//     std::vector<double> out;
//     out.reserve(actualPlayers);
//     for (unsigned i = 0; i < actualPlayers && i < MAX_PLAYERS; ++i) {
//         out.push_back(res.equity[i]);
//         if (debug) {
//             std::cout << "Equity player " << (i+1) << " = " << res.equity[i] << "\n";
//         }
//     }

//     return out;
// }

// // Python bindings
// PYBIND11_MODULE(equity_wrapper, m) {
//     m.doc() = "OMPEval equity wrapper (Omaha/generic)";

//     m.def("compute_equity", &compute_equity,
//           py::arg("hands"),
//           py::arg("board") = std::vector<std::string>(),
//           py::arg("debug") = false,
//           "Compute equities for given hands and (optional) board. "
//           "hands: list of players, each player is list of card strings (e.g. [['As','Kd','Qh','Jd'], ...])\n"
//           "board: list of board card strings (0..5 cards)\n"
//           "debug: print debug info to stdout");
// }
