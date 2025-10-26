export const RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];
export const SUITS = ["s", "h", "d", "c"];

export function makeDeck() {
    const deck = [];
    for (const r of RANKS) {
        for (const s of SUITS) {
            deck.push(`${r}${s}`);
        }
    }
    return deck;
}

export function shuffleDeck(inputDeck, rng=Math.random) {
    const deck = inputDeck.slice();
    for (let i = deck.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [deck[i], deck[j]] = [deck[j], deck[i]];
    }
    return deck;
}

export function deal(deck, n=1) {
    if (n < 0) throw new Error("n must be >= 0");
    return {
        dealt: deck.slice(0, n),
        deck: deck.slice(n),
    };
}