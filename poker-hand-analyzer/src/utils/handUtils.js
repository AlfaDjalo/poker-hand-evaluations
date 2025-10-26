import { makeDeck, shuffleDeck, deal } from "./deckUtils";

export function createNewHand({ numPlayers=6, startingStack=200, numCards=4 }={}) {
    const players = {};
    for (let i=1; i<=numPlayers; i++) {
        players[i] = {
            seat: i,
            name: `Player ${i}`,
            stack: startingStack,
            hand: [],
            hasFolded: false,
            bet: 0,
            isActive: true,
        };
    }

    const deck = shuffleDeck(makeDeck());

    return {
        id: Date.now(),
        players,
        deck,
        board: [],
        pot: 0,
        street: "pre-flop",
        dealer: 1,
    };
}

export function dealHoleCards(hand, numCards=4) {
    let deck = hand.deck.slice();
    const players = { ...hand.players };

    for (const seat of Object.keys(players)) {
        // const player = players[seat];

        // if (!player || !player.isActive) continue;

        const dealt = deck.slice(0, numCards).map(card => ({ card, hidden: false }));
        deck = deck.slice(numCards);
        // deck = rest;
        // const rest = deck.slice(numCards);

        players[seat] = { ...players[seat], hand: dealt };
        // players[seat] = { ...player, hand: dealt };
    }

    return {
        // ...players,
        // hand,
        ...hand,
        players,
        deck
    };
}