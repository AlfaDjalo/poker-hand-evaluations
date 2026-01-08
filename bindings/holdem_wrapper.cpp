#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "holdem_equity.h"
#include <vector>
#include <string>

namespace py = pybind11;
using namespace omp;

// Simple wrapper that returns winner (0 or 1)
int evaluate_showdown(
    const std::vector<std::string>& hand1,
    const std::vector<std::string>& hand2,
    const std::vector<std::string>& board,
    bool debug = false)
{
    std::vector<std::vector<std::string>> hands = {hand1, hand2};
    
    HoldemEquityResults results = compute_holdem_equity(
        hands, board, true, 0, debug
    );
    
    // Return 0 if player 1 wins, 1 if player 2 wins
    // In case of tie, randomly pick (though ties should be split in equity)
    if (results.equity[0] > results.equity[1]) {
        return 0;
    } else if (results.equity[1] > results.equity[0]) {
        return 1;
    } else {
        // Tie - return 0 (could randomize if needed)
        return 0;
    }
}

PYBIND11_MODULE(holdem_wrapper, m) {
    m.doc() = "Hold'em hand evaluator using OMPEval";

    m.def("evaluate_showdown", &evaluate_showdown,
          py::arg("hand1"),
          py::arg("hand2"),
          py::arg("board"),
          py::arg("debug") = false,
          "Evaluate Hold'em showdown between two hands.\n\n"
          "Args:\n"
          "  hand1: List of 2 card strings (e.g. ['As','Kd'])\n"
          "  hand2: List of 2 card strings\n"
          "  board: List of 5 board card strings\n"
          "  debug: Print debug info\n\n"
          "Returns:\n"
          "  0 if hand1 wins, 1 if hand2 wins");
}