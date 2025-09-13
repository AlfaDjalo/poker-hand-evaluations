import React, { useState } from "react";
import CardSelector from './CardSelector';
import PokerTable from './PokerTable';

const EquityCalculator = () => {
  // Players with empty hands
  const [players, setPlayers] = useState({
    1: { name: "Alice", chips: 1500, hand: [] },
    2: { name: "Bob", chips: 1200, hand: [] },
    3: { name: "Charlie", chips: 2000, hand: [] },
  });

  const [boardCards, setBoardCards] = useState([]);

  // Track where to send selected cards.
  const [activeTarget, setActiveTarget] = useState(null);
  // Values could be { type: "player", seat: 1 } or { type: "board" }

  const handleCardSelected = (card) => {
    console.log(activeTarget)
    if (!activeTarget) return;

    if (activeTarget.type === "player") {
      console.log("Selecting player card")
      setPlayers((prev) => {
        const updated = { ...prev };
        const hand = updated[activeTarget.seat].hand;
        if (hand.length < 5) {
          updated[activeTarget.seat] = {
            ...updated[activeTarget.seat],
            hand: [...hand, { card, hidden:false }]
          };
        }
        return updated;
      });
    }

    if (activeTarget.type === "board") {
      console.log("Selecting board card")
      setBoardCards((prev) => {
        if (prev.length < 5) {
          return [...prev, { card, hidden: false }];
        }
        return prev;
      });
    }
  };

  return (
    <div className="equity-calculator">
      <h2>Equity Calculator</h2>

      <CardSelector onSelectCard={handleCardSelected} />

      <PokerTable 
        players={players}
        boardCards={boardCards}
        dealerSeat={1}
        onSeatClick={(seat) => setActiveTarget({ type: "player", seat })}
        onBoardClick={() => setActiveTarget({ type: "board" })}
      >
      </PokerTable>
    </div>
  );
};

export default EquityCalculator;
