import React, { useState } from 'react';

import PokerTable from "./PokerTable";
// import { makeDeck, shuffleDeck, deal } from "../utils/deckUtils";
import { createNewHand, dealHoleCards } from "../utils/handUtils";

const GameSimulator = () =>
{
    const [hand, setHand] = useState(null);

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

    // const handleDealBoard = () => {
    //     if (!hand) return;
    //     const dealt = dealBoardCards(hand, 3);
    //     // setHand(dealt);
    //     // console.log("Dealt hand: ", dealt.players.map(p => ({ seat: p.seat, hand: p.hand })));
    // };

    return (
        <div Style={{ padding: 16 }}>
            <h3> Game Simulator </h3>   
            <button onClick={ handleNewHand }>Create New Hand</button>
            <button onClick={ handleDeal }>Deal Hole Cards</button>

            {/* <PokerTable
                players={hand.players}
                boardCards={hand.board}
                dealerSeat={1}
            /> */}

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

            {/* <div style={{ marginTop: 16 }}>
                <pre style={{ maxHeight:300, overflow: "auto" }}>
                    {hand && hand.players ? JSON.stringify({
                        players:  Object.values(hand.players).map(p => ({ seat: p.seat, hand: p.hand, stack: p.stack })),
                        // players: hand.players.map(p => ({ seat: p.seat, hand: p.hand, stack: p.stack })),
                        deckRemaining: hand.deck.length,
                        board: hand.board,
                        pot: hand.pot,
                        street: hand.street,
                    }, null, 2) : "No hand yet."}
                </pre>
            </div> */}
        </div>
    );
};

export default GameSimulator;