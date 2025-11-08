import { makeDeck, shuffleDeck } from "./deckUtils";

export function startNewHand({ players, blinds, numHoleCards, numBoardCards }) {
  let hand = createNewHand({ playerInput: players, numCards: numHoleCards });
  hand = dealHoleCards(hand, numHoleCards);
  hand = payBlinds(hand, blinds);
  hand.nextToAct = getNextActiveSeat(hand.players, hand.dealerSeat);
  hand.availableActions = getAvailableActions(hand, hand.nextToAct);
  return hand;
}

export function createNewHand({ playerInput, numCards=4 }={}) {
    const players = {};
    for (const [seat, player] of Object.entries(playerInput)) {
        players[seat] = {
            ...player,
            hasFolded: false,
            hand: [],
            contributionCurrentStreet: 0,
            contributionHand: 0,
        };
    }    

    const deck = shuffleDeck(makeDeck());
    
    console.log("Players in new hand:", players);

    return {
        id: Date.now(),
        players,
        deck,
        board: [],
        betting: {
            currentBet: 0,
            lastAggressor: 1,
            minBet: 0
        },
        pot: 0,
        street: 0,
        dealer: 1,
    };
}

export function dealHoleCards(hand, numCards=4) {
    let deck = hand.deck.slice();
    const players = { ...hand.players };

    for (const seat of Object.keys(players)) {
        const dealt = deck.slice(0, numCards).map(card => ({ card, hidden: false }));
        deck = deck.slice(numCards);

        players[seat] = { ...players[seat], hand: dealt };
    }

    console.log("Players dealt hold cards:", players);

    return {
        ...hand,
        players,
        deck
    };
}

export function payBlinds(hand, blinds=[1,2], ante=0) {
    // Players to dealer left pay blinds.
    const players = { ...hand.players };
    let pot = hand.pot;
    let betting = hand.betting;
    let currentSeat = hand.dealer;
    
    blinds.forEach((blindAmount) => {
        let seat = getNextActiveSeat(players, currentSeat);
        players[seat].stack -= blindAmount;
        players[seat].contributionCurrentStreet += blindAmount;
        players[seat].contributionHand += blindAmount;
        currentSeat = seat;
        pot += blindAmount;
        betting.currentBet = blindAmount;
        betting.lastAggressor = seat;
        betting.minBet = blindAmount;
    });

    const nextToAct = getNextActiveSeat(players, currentSeat);
    betting.lastAggressor = nextToAct;
    console.log("Next to act: ", nextToAct);

    for (const seat of Object.keys(players)) {
        const player = players[seat];
        if (player.isActive) {
            player.stack -= ante;
            player.contributionCurrentStreet += ante;
            player.contributionHand += ante;
            pot += ante;
        };
    }

    console.log("Players after paying blinds:", players);

    return {
        ...hand,
        players,
        betting,
        pot, 
        nextToAct
    };
}

export function placeBet(hand, actionPlayer, betAmount) {
    const players = { ...hand.players };
    const player = players[actionPlayer];
    const betting = { ...hand.betting};
    let pot = hand.pot;

    if (betAmount > player.stack) {
        console.log("Cannot bet more than stack, no bet made");
        return hand;
    }
    
    const { minBet, maxBet } = getBetLimits(hand, actionPlayer);
    
    if (betAmount < minBet) {
        return hand;
    }
    
    if (betAmount > maxBet) {
        betAmount = maxBet;
    }
    
    console.log("Player betting ", betAmount);
    
    player.stack -= betAmount;
    player.contributionCurrentStreet += betAmount;
    player.contributionHand += betAmount;
    pot += betAmount;
    console.log("Player ", actionPlayer, "bet ", betAmount);

    betting.currentBet = betAmount;
    betting.lastAggressor = actionPlayer;

    players[actionPlayer] = player;

    return {
        ...hand,
        players,
        betting,
        pot
    };
}

export function callBet(hand, actionPlayer) {
    const players = { ...hand.players };
    const player = players[actionPlayer];
    const betting = { ...hand.betting};
    const betAmount = betting.currentBet;
    let pot = hand.pot;

    console.log("Player calling ", betAmount);

    let callAmount = betAmount - player.contributionCurrentStreet;
    if (callAmount <= 0) return hand;

    if (callAmount > player.stack) {
        callAmount = player.stack;
    }

    console.log("Player calling ", callAmount);

    player.stack -= callAmount;
    player.contributionCurrentStreet += callAmount;
    player.contributionHand += callAmount;
    pot += callAmount;
    console.log("Player ", actionPlayer, "called extra ", callAmount);

    players[actionPlayer] = player;

    return {
        ...hand,
        players,
        pot
    };
}

export function foldPlayer(hand, actionPlayer) {
    const players = { ...hand.players };

    players[actionPlayer].hasFolded = true;
    console.log("Player ", actionPlayer, "folded ");

    return {
        ...hand,
        players,
    };
}

export function check(hand, actionPlayer) {

    console.log("Player ", actionPlayer, "checked ");

    return {
        ...hand
    };
}

export function isBettingRoundComplete(hand, actionPlayer) {
    const players = { ...hand.players };
    const betting = { ...hand.betting };

    const activePlayers = Object.values(players).filter(p => p.isActive);

    console.log("Checking if betting round is complete")
    console.log(activePlayers);

    return actionPlayer = betting.lastAggressor;
    // return activePlayers.every(p => p.hasFolded || p.contributionCurrentStreet === betting.currentBet || p.stack === 0);
}

export function endStreet(hand) {
    const players = { ...hand.players };
   
    for (const seat of Object.keys(players)) {
        const player = players[seat];
        player.contributionCurrentStreet = 0;
    }

    let betting = hand.betting;
    betting.currentBet = 0;
    betting.lastAggressor = hand.dealer;

    const street = hand.street + 1;
    const nextToAct = getNextActiveSeat(players, hand.dealer);

    return {
        ...hand,
        players,
        betting,
        street,
        nextToAct
    }
}



export function getNextActiveSeat(players, startSeat) {
    const seats = Object.keys(players)
        .map(Number)
        .sort((a, b) => a - b);

    const startIndex = seats.indexOf(startSeat);
    const numSeats = seats.length;

    for (let offset=1; offset<=numSeats; offset++) {
        const nextSeat = seats[(startIndex + offset) % numSeats];
        if (players[nextSeat]?.isActive && !players[nextSeat].hasFolded) {
            return nextSeat;
        }
    }
}

function getBetLimits(hand, actionPlayer, structure="no-limit") {
    const player = hand.players[actionPlayer];
    const betting = hand.betting;
    const pot = hand.pot;

    const callAmount = betting.currentBet - player.contributionCurrentStreet;
    let minBet, maxBet;

    switch (structure) {
        case "fixed-limit":
            minBet = betting["minBet"];
            maxBet = minBet;
            break;
        case "pot-limit":
            minBet = betting["minBet"];
            maxBet = pot + callAmount;
            break;
        case "no-limit":
        default:
            minBet = betting["minBet"];
            maxBet = player.stack;
            break;
    }

    return { minBet, maxBet };
}

export function getAvailableActions(hand, actionPlayer) {
    console.log("Players: ", hand.players);
    console.log("Action player: ", actionPlayer);
    const player = hand.players[actionPlayer];
    if (!player || player.hasFolded) return [];

    const currentBet = hand.betting.currentBet;
    const playerContribution = player.contributionCurrentStreet;

    const actions = ["fold"];

    const canCall = playerContribution < currentBet && player.stack > 0;
    const canCheck = (playerContribution === currentBet); 
    const canBetOrRaise = player.stack > currentBet;

    if (canCall) actions.push("call");
    if (canCheck) actions.push("check");
    if (canBetOrRaise) actions.push("bet");

    return actions;
}