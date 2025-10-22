// equity_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <omp/EquityCalculator.h>   // OMPEval header (adjust include path)
#include <omp/CardRange.h>          // for range parsing if needed
#include <string>
#include <vector>

namespace py = pybind11;
using namespace omp;

// helper: convert "As" -> uint64 mask (OMPEval has CardRange helpers)
uint64_t mask_from_strings(const std::vector<std::string>& cards) {
    uint64_t mask = 0;
    for (const auto &c : cards) {
        mask |= CardRange::getCardMask(c);
    }
    return mask;
}

// simple API: player_hands = vector<vector<string>> of exact combos (like ["As","Ks"])
py::dict compute_equity_explicit(const std::vector<std::vector<std::string>>& player_hands,
                                 const std::vector<std::string>& board_cards,
                                 const std::string& mode = "enumerate",
                                 int montecarlo_samples = 100000,
                                 unsigned threads = 0)
{
    // Convert explicit hands into CardRange-like entries or direct masks
    // OMPEval's EquityCalculator can accept vector<CardRange> or vector<string> ranges
    std::vector<std::string> ranges;
    for (const auto &hand : player_hands) {
        // join two-card hand into canonical like "AsKs"
        if (hand.size() == 2) {
            ranges.push_back(hand[0] + hand[1]);
        } else {
            throw std::invalid_argument("Each hand must be two cards for Omaha/Hold'em use-cases");
        }
    }

    EquityCalculator eq;
    // Build board / dead masks
    uint64_t boardMask = mask_from_strings(board_cards);
    uint64_t deadMask = boardMask; // plus any used cards in hands - OMPEval accepts dead mask param

    // run: OMPEval's equity calculator has start(ranges, boardMask, deadMask, enumerateFlag, stderrMargin, callback,...)
    if (mode == "enumerate") {
        eq.start(ranges, boardMask, deadMask, true /*enumerate*/, 0.0, nullptr, 1.0, threads);
    } else {
        double stdErr = 1e-3; // stop when std error small
        eq.start(ranges, boardMask, deadMask, false /*montecarlo*/, stdErr, nullptr, 1.0, threads);
    }
    eq.wait();
    auto results = eq.getResults();

    py::dict out;
    // results.equity is vector<double>
    py::list equities;
    for (double e : results.equity) equities.append(e);
    out["equity"] = equities;
    out["wins"] = py::list();
    for (auto w : results.wins) py::cast(out["wins"]).attr("append")(w);
    out["hands_evaluated"] = (long long)results.hands;
    out["time"] = results.time;
    return out;
}

PYBIND11_MODULE(ompeval_cpp, m) {
    m.doc() = "OMPEval wrapper";
    m.def("compute_equity_explicit", &compute_equity_explicit,
          py::arg("player_hands"), py::arg("board_cards"),
          py::arg("mode") = "enumerate",
          py::arg("montecarlo_samples") = 100000,
          py::arg("threads") = 0);
}
