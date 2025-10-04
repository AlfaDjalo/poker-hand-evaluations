#include <iostream>
#include <vector>
#include <string>

// Super trivial evaluator: returns the highest rank in a 5-card hand
// Ranks are passed in as ints: 2-14 (where Ace = 14)
int evaluate_highcard(const std::vector<int>& hand) {
    int maxRank = 0;
    for (int rank : hand) {
        if (rank > maxRank) {
            maxRank = rank;
        }
    }
    return maxRank;
}

int main() {
    // Example: test with a 5-card hand
    std::vector<int> hand = {2, 5, 9, 11, 14}; // Ace high
    int score = evaluate_highcard(hand);

    std::cout << "Highest card rank: " << score << std::endl;

    return 0;
}
