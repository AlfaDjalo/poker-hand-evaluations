import { deal } from "./deckUtils";

export const dealBoardCards = (hand, numCards) => {
    const newBoard = [...hand.board];
    let newHand = { ...hand };
    let deck = newHand.deck.slice();

    const dealt = deck.slice(0, numCards);
    deck = deck.slice(numCards);
    
    const dealtObjects = dealt.map(card => ({ card, hidden: false }));
    newBoard.push(...dealtObjects);
    // newHand.board = [...(newHand.board || []), ...dealt];
    // newHand.board = [...newHand.board, ...dealt];

    return {
        ...hand,
        board: newBoard,
        deck
    }
    // return newHand;
}