import React, { useState, useEffect } from "react";
import CardSelector from './CardSelector';
import PokerTable from './PokerTable';
import { evaluateHand } from "../api";
import "../css/EquityCalculator.css";
import { DndContext, PointerSensor, useSensor, useSensors } from "@dnd-kit/core";

const EquityCalculator = () => {
  // Players with empty hands
  const [players, setPlayers] = useState({
    1: { name: "Alice", chips: 1500, hand: [null, null, null, null] },
    2: { name: "Bob", chips: 1200, hand: [null, null, null, null] },
    3: { name: "Charlie", chips: 2000, hand: [null, null, null, null] },
  });
  
  const [boardCards, setBoardCards] = useState([null, null, null, null, null]);
    
  const [result, setResult] = useState(null);
  const [selectedCards, setSelectedCards] = useState([]);
  const [activeTarget, setActiveTarget] = useState(null);
  const [calculating, setCalculating] = useState(false);

  // activeTarget values:
// { type: "player", seat: 1, slot: 0 } - specific hand slot
// { type: "player", seat: 1 } - any available slot in player's hand
// { type: "board", slot: 0 } - specific board slot
// { type: "board" } - any available board slot

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: { distance: 100 }, // only start dragging after 5px movement
    })
  );

  async function runEvaluation() {
    setCalculating(true);

    // Clear equities immediately on new run
    setPlayers(prev => {
      const updated = { ...prev };
      Object.keys(updated).forEach(seat => {
        updated[seat] = { ...updated[seat], equity: undefined };
      });
      return updated;
    });

    try {
      const playerHands = Object.values(players).map(p =>
        p.hand.map(c => c?.card)
      );
      const board = boardCards.map(c => c?.card);

      const res = await evaluateHand(playerHands, board);
      setResult(res);

      const seats = Object.keys(players);
      setPlayers(prev => {
        const updated = { ...prev };
        seats.forEach((seat, idx) => {
          updated[seat] = {
            ...updated[seat],
            equity: res.equities[idx] * 100
          };
        });
        return updated;
      });
    } catch (err) {
      console.error("Evaluation failed:", err);
    } finally {
      setCalculating(false);
    }
  }

  const clearEquities = () => {
    setPlayers(prev => {
      const updated = { ...prev };
      Object.keys(updated).forEach(seat => {
        updated[seat] = { ...updated[seat], equity: undefined };
      });
      return updated;
    });
    setResult(null);
  };

  useEffect(() => {
    // Whenever players' hands or board changes, clear equities
    // setPlayers(prev => {
    //   const updated = { ...prev };
    //   Object.keys(updated).forEach(seat => {
    //     updated[seat] = { ...updated[seat], equity: undefined };
    //   });
    //   return updated;
    // });
    setResult(null); // also clear previous result if you show it
  }, [players, boardCards]);

  const handleCardRemoval = (type, identifier, cardIndex) => {
    clearEquities();
    if (type === "player") {
      setPlayers(prev => {
        const updated = { ...prev };
        const newHand = [...updated[identifier].hand];
        newHand.splice(cardIndex, 1); // Remove card at index
        updated[identifier] = {
          ...updated[identifier],
          hand: newHand
        };
        return updated;
      });
    }

    if (type === "board") {
      setBoardCards(prev => {
        const newBoard = [...prev];
        newBoard[cardIndex] = null; // Set slot to empty
        return newBoard;
      });
    }
  };

  const handleCardSelected = (card) => {
    if (!activeTarget) return;

    clearEquities();
    if (activeTarget.type === "player") {
      setPlayers((prev) => {
        const updated = { ...prev };
        const hand = [...updated[activeTarget.seat].hand];

        if (activeTarget.slot != null) {
          // Target specific slot
          hand[activeTarget.slot] = { card, hidden: false };
        } else {
          // Target first available empty slot (null)
          const emptyIndex = hand.findIndex(c => !c);
          if (emptyIndex !== -1) {
            hand[emptyIndex] = { card, hidden: false };
          }
        }

        updated[activeTarget.seat] = {
          ...updated[activeTarget.seat],
          hand
        };

        // ✅ Calculate next slot from the updated hand
        const nextEmptyIndex = hand.findIndex(c => !c);
        setActiveTarget(nextEmptyIndex !== -1 
          ? { type: "player", seat: activeTarget.seat, slot: nextEmptyIndex } 
          : null
        );

        return updated;
      });

      // Move activeTarget to next available slot
      setActiveTarget(prev => {
        const seat = prev.seat;
        const nextIndex = players[seat].hand.findIndex(c => !c);
        return nextIndex !== -1 ? { type: "player", seat, slot: nextIndex } : null;
      });
    }

    if (activeTarget.type === "board") {
      setBoardCards((prev) => {
        const newBoard = [...prev];
        const targetIndex = activeTarget.slot != null 
          ? activeTarget.slot 
          : newBoard.findIndex((c) => !c);
          
        if (targetIndex >= 0) {
          newBoard[targetIndex] = { card, hidden: false };
        }
        return newBoard;
      });
    }
  };

  const handleCardDropToPlayer = (card, seat, slotIndex) => {
    clearEquities();

    setPlayers(prev => {
      const updated = { ...prev };
      const hand = [...updated[seat].hand];
      hand[slotIndex] = { card, hidden: false };
      updated[seat] = { ...updated[seat], hand };
      return updated;
    });
  };

  const handleCardDropToBoard = (card, slotIndex) => {
    clearEquities();

    setBoardCards(prev => {
      const newBoard = [...prev];
      newBoard[slotIndex] = { card, hidden: false };
      return newBoard;
    });
  };

  const handleDragEnd = (event) => {
    const { active, over } = event;
    console.log("DragEnd event:", event);
    if (!over) {
      console.log("Dropped outside any droppable");
      return;
    }

    if (!over) return; // dropped outside any droppable

    const draggedCard = active.data.current?.card; // the card object
    const targetId = over.id; // droppable id
    console.log("Dragged card:", draggedCard, "Target ID:", targetId);

    if (!draggedCard || !targetId) {
      console.log("No dragged card or target ID");
      return;
    }

    // --- Player slot drop ---
    if (targetId.startsWith("player-") && targetId.includes("slot")) {
      // format: "player-{seat}-slot-{slotIndex}"
      const [, seatStr, , slotStr] = targetId.split("-");
      const seat = Number(seatStr);
      const slotIndex = Number(slotStr);
      handleCardDropToPlayer(draggedCard, seat, slotIndex);
      return;
    }

    // --- Player area drop (any slot in player hand) ---
    if (targetId.startsWith("player-area")) {
      // format: "player-area-{seat}"
      const [, , seatStr] = targetId.split("-");
      const seat = Number(seatStr);

      const hand = players[seat].hand;
      const emptyIndex = hand.findIndex((c) => !c);
      if (emptyIndex !== -1) {
        handleCardDropToPlayer(draggedCard, seat, emptyIndex);
      }
      return;
    }

    // --- Board slot drop ---
    if (targetId.startsWith("board-slot")) {
      // format: "board-slot-{index}"
      const [, , slotStr] = targetId.split("-");
      const slotIndex = Number(slotStr);
      handleCardDropToBoard(draggedCard, slotIndex);
      return;
    }

    // --- Board area drop (no specific slot) ---
    if (targetId === "board-area") {
      const emptyIndex = boardCards.findIndex((c) => !c);
      if (emptyIndex !== -1) {
        handleCardDropToBoard(draggedCard, emptyIndex);
      }
      return;
    }
  };




  // const handleDragEnd = (event) => {
  //   const { active, over } = event;

  //   if (!over) return; // dropped outside any droppable

  //   const draggedCard = active.data.current?.card; // the card object
  //   const targetId = over.id; // droppable id

  //   if (!draggedCard || !targetId) return;

  //   // --- Player slot drop ---
  //   if (targetId.startsWith("player") && targetId.includes("slot")) {
  //     // format: "player-{seat}-slot-{slotIndex}"
  //     const [, seatStr, , slotStr] = targetId.split("-");
  //     const seat = Number(seatStr);
  //     const slotIndex = Number(slotStr);
  //     handleCardDropToPlayer(draggedCard, seat, slotIndex);
  //     return;
  //   }

  //   // --- Player area drop (no specific slot) ---
  //   if (targetId.startsWith("player-area")) {
  //     // format: "player-area-{seat}"
  //     const [, , seatStr] = targetId.split("-");
  //     const seat = Number(seatStr);

  //     // find first empty slot in this player's hand
  //     const hand = players[seat].hand;
  //     const emptyIndex = hand.findIndex((c) => !c);
  //     if (emptyIndex !== -1) {
  //       handleCardDropToPlayer(draggedCard, seat, emptyIndex);
  //     }
  //     return;
  //   }

  //   // --- Board slot drop ---
  //   if (targetId.startsWith("board-slot")) {
  //     // format: "board-slot-{index}"
  //     const [, , slotStr] = targetId.split("-");
  //     const slotIndex = Number(slotStr);
  //     handleCardDropToBoard(draggedCard, slotIndex);
  //     return;
  //   }

  //   // --- Board area drop (no specific slot) ---
  //   if (targetId === "board-area") {
  //     const emptyIndex = boardCards.findIndex((c) => !c);
  //     if (emptyIndex !== -1) {
  //       handleCardDropToBoard(draggedCard, emptyIndex);
  //     }
  //     return;
  //   }
  // };

  // const handleDragEnd = (event) => {
  //   const { active, over } = event;

  //   if (!over) return; // dropped outside any droppable

  //   const draggedCard = active.data.current?.card; // the card object
  //   const targetId = over.id; // droppable id, e.g., "player-1-slot-2" or "board-slot-0"

  //   if (!draggedCard || !targetId) return;

  //   if (targetId.startsWith("player")) {
  //     // player slot id format: "player-{seat}-slot-{slotIndex}"
  //     const [, seatStr, , slotStr] = targetId.split("-");
  //     const seat = Number(seatStr);
  //     const slotIndex = Number(slotStr);
  //     handleCardDropToPlayer(draggedCard, seat, slotIndex);
  //   }

  //   if (targetId.startsWith("board")) {
  //     // board slot id format: "board-slot-{index}"
  //     const [, , slotStr] = targetId.split("-");
  //     const slotIndex = Number(slotStr);
  //     handleCardDropToBoard(draggedCard, slotIndex);
  //   }
  // };


  // Collect all used cards from all player hands and the board
  const usedCards = [
    ...Object.values(players)
      .flatMap(player => player.hand.filter(Boolean).map(c => c.card)), 
    ...boardCards.filter(Boolean).map(c => c.card)
  ];


  return (
    <DndContext
      sensors={sensors}                 // <-- add sensors here
      onDragStart={event => {
        console.log("Drag started:", event);
      }}
      onDragEnd={event => {
        console.log("Drag ended:", event);
        handleDragEnd(event)
      }}
    >

      <div className="equity-calculator">
        <h2>Equity Calculator</h2>

        <CardSelector
          onSelectCard={handleCardSelected}
          usedCards={usedCards}
        />

        {/* <CardSelector onSelectCard={handleCardSelected} /> */}

        <div className="equity-calculator">
          <button onClick={runEvaluation} disabled={calculating}>
            {calculating ? "Calculating..." : "Evaluate"}
          </button>

          {calculating && (
            <div className="overlay">
              <div className="spinner" />
              <p>Calculating equities...</p>
            </div>
          )}
        </div>

        <PokerTable
          players={players}
          boardCards={boardCards}
          dealerSeat={1}
          activeTarget={activeTarget}
          onSeatClick={(seat) => setActiveTarget({ type: "player", seat })}
          onPlayerCardClick={(seat, cardIndex) => handleCardRemoval("player", seat, cardIndex)}
          onPlayerSlotClick={(seat, slotIndex) => setActiveTarget({ type: "player", seat, slot: slotIndex })}
          onBoardSlotClick={(slotIndex) => setActiveTarget({ type: "board", slot: slotIndex })}
          onBoardCardClick={(cardIndex) => handleCardRemoval("board", "board", cardIndex)}
          onBoardAreaClick={() => setActiveTarget({ type: "board" })}
          loading={calculating}
        />
      </div>
    </DndContext>
  );
};

export default EquityCalculator;
