import React, { useState } from 'react';

import PokerTable from "./PokerTable";
import { createNewHand, dealHoleCards } from "../utils/handUtils";
import { dealBoardCards } from "../utils/boardUtils";
import { evaluateShowdown } from "../utils/api";

const GameSimulator = () =>
{
    const [hand, setHand] = useState(null);
    const [showdownResult, setShowdownResult] = useState(null);

    const numBoardCards = [0, 3, 1, 1];

    const handleNewHand = () => {
        const newHand = createNewHand({ numPlayers: 6, startingStack: 200, numCards: 4 });
        setHand(newHand);
    };

    const handleDeal = () => {
        if (!hand) return;
        const dealt = dealHoleCards(hand, 4);
        setHand(dealt);
        console.log("Dealt hand: ", Object.values(dealt.players).map(p => ({ seat: p.seat, hand: p.hand })));
    };

    const handleDealBoard = () => {
        if (!hand) return;
        const nextStreet = hand.street + 1;
        if (nextStreet > numBoardCards.length - 1) return;
        const numCards = numBoardCards[nextStreet] || 0; 
        const newHand = { ...hand, street: nextStreet };
        const updated = dealBoardCards(newHand, numCards);
        setHand({ ...updated });

        console.log("Updated board: ", Object.values(updated.board));
    }

    const handleShowdown = async () => {
        if (!hand) return;
        if (hand.street < numBoardCards.length - 1) return;

        const playerHands = Object.values(hand.players).map(p => ( p.hand.map(c => c.card) ));
        const board = hand.board.map(c => c.card);

        console.log("Showdown");

        try {
            const result = await evaluateShowdown(playerHands, board);

            console.log(result);

            const seats = Object.keys(hand.players);
            const updatedPlayers = { ...hand.players };
            seats.forEach((seat, idx) => {
                updatedPlayers[seat] = {
                    ...updatedPlayers[seat],
                    equity: result.equities[idx] * 100
                };
            });

            setHand(prev => ({ ...prev, players: updatedPlayers }));
            setShowdownResult(result.equities);
            console.log("Showdown result: ", result);
        } catch (err) {
            console.error(err);
        }
    };

    return (
        <div style={{ padding: 16 }}>
            <h3> Game Simulator </h3>   
            <button onClick={ handleNewHand }>Create New Hand</button>
            <button onClick={ handleDeal }>Deal Hole Cards</button>
            <button onClick={ handleDealBoard }>Deal Next Street</button>
            <button onClick={ handleShowdown }>Showdown</button>

            { console.log("Rendering PokerTable, boardCards =", hand?.board)}
            
            <PokerTable
            players={hand ? hand.players : {}}
            boardCards={hand ? hand.board : []}
            dealerSeat={1}              // or hand?.dealerButton if you add it later
            activeTarget={null}         // placeholder until you wire DnD targeting
            onSeatClick={(seatNum) => console.log("Seat clicked:", seatNum)}
            onPlayerCardClick={(seatNum, cardIndex) =>
                console.log(`Clicked player ${seatNum} card ${cardIndex}`)
            }
            onPlayerSlotClick={(seatNum, slotIndex) =>
                console.log(`Clicked slot ${slotIndex} for player ${seatNum}`)
            }
            onBoardCardClick={(index) =>
                console.log(`Clicked board card ${index}`)
            }
            onBoardSlotClick={(index) =>
                console.log(`Clicked board slot ${index}`)
            }
            onBoardAreaClick={() => console.log("Board area clicked")}
            loading={false}
            />

            <div>
                {showdownResult && Object.keys(hand.players).map((seat, idx) => (
                    <div key={seat} style={{ fontWeight: showdownResult[idx] > 0 ? "bold" : "normal" }}>
                        Player {seat}: {hand.players[seat].hand.map(c => c.card).join(" ")}
                        {showdownResult[idx] === 1 ? " (Winner)" : showdownResult[idx] > 0 ? " (Tie)" : ""}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default GameSimulator;