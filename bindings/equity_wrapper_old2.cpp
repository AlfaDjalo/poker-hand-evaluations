#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <iostream>
#include "EquityCalculator.h"
#include "CardRange.h"
#include "HandEvaluator.h"

namespace py = pybind11;
using namespace omp;

// Helper: convert card strings like "Ts" -> "TS"
std::string normalize_card(const std::string& card) {
    std::string c = card;
    std::transform(c.begin(), c.end(), c.begin(), ::toupper);
    return c;
}

// Helper: flatten vector of cards into a single CardRange string
std::string make_combo_string(const std::vector<std::string>& cards) {
    std::string combo;
    for (const auto& card : cards) {
        if (!card.empty()) {
            std::string c = normalize_card(card);
            combo += c; // no space separator (OMPEval expects contiguous)
        }
    }
    return combo;
}

std::vector<double> compute_equity(
    const std::vector<std::vector<std::string>>& hands,
    const std::vector<std::string>& board)
{
    EquityCalculator calc;
    std::vector<CardRange> ranges;
    ranges.reserve(hands.size());

    // Convert Python hands -> OMPEval CardRanges
    for (const auto& h : hands) {
        std::string combo = make_combo_string(h);
        if (combo.empty()) {
            std::cerr << "[Warning] Empty hand detected, skipping.\n";
            continue;
        }
        try {
            ranges.emplace_back(combo);
        } catch (const std::exception& e) {
            std::cerr << "[Error] Invalid CardRange for hand: " << combo
                      << " (" << e.what() << ")\n";
            ranges.emplace_back(""); // placeholder empty range
        }
    }

    // Convert board -> mask
    uint64_t boardMask = 0;
    for (const auto& c : board) {
        if (!c.empty()) {
            std::string card = normalize_card(c);
            try {
                boardMask |= CardRange::getCardMask(card);
            } catch (const std::exception& e) {
                std::cerr << "[Error] Invalid board card: " << card
                          << " (" << e.what() << ")\n";
            }
        }
    }

    // Sanity print
    std::cerr << "[Debug] Hands parsed: " << ranges.size() << "\n";
    std::cerr << "[Debug] Board mask: " << boardMask << "\n";

    // Run the calculator
    calc.start(ranges, boardMask, 0ULL, true, 0.0001);
    calc.wait();

    auto res = calc.getResults();

    // OMPEval’s EquityCalculator::Results::equity is typically a fixed array (6)
    // but we only return as many results as hands were provided.
    std::vector<double> equities;
    equities.reserve(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i)
        equities.push_back(res.equity[i]);

    return equities;
}

PYBIND11_MODULE(equity_wrapper, m) {
    m.doc() = "Python bindings to OMPEval EquityCalculator";
    m.def("compute_equity", &compute_equity, "Compute hand equities");
}
