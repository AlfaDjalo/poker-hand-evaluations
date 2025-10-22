#include <iostream>
#include "omp/EquityCalculator.h"
#include "omp/CardRange.h"

int main() {
    omp::EquityCalculator calc;
    std::vector<omp::CardRange> hands = {
        omp::CardRange("AhKh"),
        omp::CardRange("QcQs")
    };

    bool ok = calc.start(hands, 0, 0, true, 0.0001);  // enumerateAll=true
    if (!ok) {
        std::cout << "Start failed" << std::endl;
        return 1;
    }

    calc.wait();  // 🔹 Let threads complete

    auto res = calc.getResults();

    std::cout << "Finished: " << res.finished << std::endl;
    std::cout << "Players: " << res.players << std::endl;
    std::cout << "Equities: " << res.equity[0] << " " << res.equity[1] << std::endl;
    std::cout << "Wins: " << res.wins[0] << " " << res.wins[1] << std::endl;
    std::cout << "Ties: " << res.ties[0] << " " << res.ties[1] << std::endl;
    std::cout << "Hands: " << res.hands << std::endl;

    return 0;
}
