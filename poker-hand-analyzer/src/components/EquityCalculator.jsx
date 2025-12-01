import React, { useState, useEffect } from "react";
import CardSelector from './CardSelector';
import PokerTable from './PokerTable';
import { evaluateHand } from "../utils/api";
import "../css/EquityCalculator.css";
import { DndContext, PointerSensor, useSensor, useSensors } from "@dnd-kit/core";
import { usePokerDnD } from "../hooks/usePokerDnD"

const EquityCalculator = () => {
  // Players with empty hands
  const [players, setPlayers] = useState({
    1: { name: "Alice", stack: 1500, hand: [null, null, null, null] },
    2: { name: "Bob", stack: 1200, hand: [null, null, null, null] },
    3: { name: "Charlie", stack: 2000, hand: [null, null, null, null] },
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
        newHand[cardIndex] = null;
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

    const { handleDragStart, handleDragEnd } = usePokerDnD({
      players,
      setPlayers,
      boardCards,
      setBoardCards,
      onCardMoved: clearEquities
    });

  // const handleDragEnd = (event) => {
  //   const { active, over } = event;
  //   console.log("DragEnd event:", event);
  //   if (!over) {
  //     console.log("Dropped outside any droppable");
  //     return;
  //   }

  //   const draggedCard = active.data.current?.card;
  //   const sourceData = active.data.current;
    
  //   const targetData = over.data.current;
  //   const targetId = over.id;
  //   if (!draggedCard || !targetId) {
  //     console.log("No dragged card or target ID");
  //     return;
  //   }

  //   console.log("Dragged card:", draggedCard, "Target ID:", targetId, "Source:", sourceData);
  //   console.log("🃏 Dropped", draggedCard, "on", targetId);
  //   console.log("Target", targetData);

  //   console.log("Source variant:", sourceData.variant);
  //   console.log("Target variant:", targetData.variant);
  //   // --- Move within component ---
  //   if (sourceData.variant === targetData.variant) {
  //     console.log("Swapping cards, variant: ", targetData.variant);
  //     if (sourceData.variant === "board") {
  //       setBoardCards(prev => {
  //         const newBoard = [...prev];
  //         // Save both cards first
  //         const sourceCard = newBoard[sourceData.index];
  //         const targetCard = newBoard[targetData.index];
  //         console.log("Swapping cards:. Target card:", targetCard, "Source card:", sourceCard);
  //         // Then swap
  //         newBoard[sourceData.index] = targetCard;
  //         newBoard[targetData.index] = sourceCard;
  //         return newBoard
  //       })
  //     } else if (sourceData.variant === "player") {
  //       const sourceSeat = sourceData.location
  //       const sourceIndex = sourceData.index
  //       const targetSeat = targetData.location
  //       const targetIndex = targetData.index
  //       const sourceCard = players[sourceSeat].hand[sourceIndex];
  //       const targetCard = players[targetSeat].hand[targetIndex];
  //       if (sourceSeat === targetSeat) {    
  //         setPlayers(prev => {
  //           const updated = { ...prev };
  //           const hand = [...updated[sourceSeat].hand];
            
  //           // Swap cards in hand
  //           [hand[sourceIndex], hand[targetIndex]] = 
  //           [hand[targetIndex], hand[sourceIndex]];
            
  //           updated[sourceSeat] = { ...updated[sourceSeat], hand };
  //           return updated;
  //         });          
  //       } else {
  //         setPlayers(prev => {
  //           const updated = { ...prev };
  //           const sourceHand = [...updated[sourceSeat].hand];
  //           const targetHand = [...updated[targetSeat].hand];
            
  //           // Swap cards in hand
  //           [sourceHand[sourceIndex], targetHand[targetIndex]] = 
  //           [targetHand[targetIndex], sourceHand[sourceIndex]];
            
  //           updated[sourceSeat] = { ...updated[sourceSeat], hand: sourceHand };
  //           updated[targetSeat] = { ...updated[targetSeat], hand: targetHand };
  //           return updated;
  //         });          

  //       }

  //     }
  //   } else if (  
  //     (sourceData.variant === "board" && targetData.variant === "player") ||
  //     (sourceData.variant === "player" && targetData.variant === "board")
  //   )
  //   {
  //     console.log("Moving between board and hand")
  //     const boardIndex = sourceData.variant === "board" 
  //       ? sourceData.index 
  //       : targetData.index;
  //     const handSeat = sourceData.variant === "player" 
  //       ? sourceData.location 
  //       : targetData.location;
  //     const handIndex = sourceData.variant === "player" 
  //       ? sourceData.index 
  //       : targetData.index;
      
  //     // console.log("boardIndex:", boardIndex)
  //     // console.log("handSeat:", handSeat)
  //     // console.log("handIndex:", handIndex)

  //     // Get both cards
  //     const boardCard = boardCards[boardIndex];
  //     const handCard = players[handSeat].hand[handIndex];
      
  //     // Update both states
  //     setBoardCards(prev => {
  //       const newBoard = [...prev];
  //       newBoard[boardIndex] = handCard;
  //       return newBoard;
  //     });
      
  //     setPlayers(prev => {
  //       const updated = { ...prev };
  //       const hand = [...updated[handSeat].hand];
  //       hand[handIndex] = boardCard;
  //       updated[handSeat] = { ...updated[handSeat], hand };
  //       return updated;
  //     });
  //   } else if (targetData.variant === "trash") {
  //     console.log("🗑️ Dropped on trash area");

  //     if (sourceData.variant === "board") {
  //       setBoardCards(prev => {
  //         const newBoard = [...prev];
  //         newBoard[sourceData.index] = null; // or ""
  //         return newBoard;
  //       });
  //     } else if (sourceData.variant === "player") {
  //       setPlayers(prev => {
  //         const updated = { ...prev };
  //         const seat = sourceData.location;
  //         // const seat = sourceData.location.seat;
  //         const hand = [...updated[seat].hand];
  //         hand[sourceData.index] = null; // or ""
  //         updated[seat] = { ...updated[seat], hand };
  //         return updated;
  //       });
  //     }
  //   }


  //   // --- Player slot drop ---
  //   // if (targetId.startsWith("player-") && targetId.includes("slot")) {
  //   if (targetData.variant === "player") {
  //     // format: "player-{seat}-slot-{slotIndex}"
  //     const seat = targetData.location
  //     const slotIndex = targetData.index
  //     // const [, seatStr, , slotStr] = targetId.split("-");
  //     // const seat = Number(seatStr);
  //     // const slotIndex = Number(slotStr);
  //     handleCardDropToPlayer(draggedCard, seat, slotIndex);
  //     return;
  //   }

  //   // --- Player area drop (any slot in player hand) ---
  //   if (targetId.startsWith("player-area")) {
  //     // format: "player-area-{seat}"
  //     const [, , seatStr] = targetId.split("-");
  //     const seat = Number(seatStr);

  //     const hand = players[seat].hand;
  //     const emptyIndex = hand.findIndex((c) => !c);
  //     if (emptyIndex !== -1) {
  //       handleCardDropToPlayer(draggedCard, seat, emptyIndex);
  //     }
  //     return;
  //   }

  //   // --- Board slot drop ---
  //   if (targetData.variant === "board") {
  //   // if (targetId.startsWith("board-slot")) {
  //     // format: "board-slot-{index}"
  //     // const board = targetData.location
  //     const slotIndex = targetData.index
  //     // const [, , slotStr] = targetId.split("-");
  //     // const slotIndex = Number(slotStr);
  //     handleCardDropToBoard(draggedCard, slotIndex);
  //     // return;
  //   }

  //   // --- Board area drop (no specific slot) ---
  //   if (targetId === "board-area" || targetId === "table-inner") {
  //     const emptyIndex = boardCards.findIndex((c) => !c);
  //     if (emptyIndex !== -1) handleDropToBoard(draggedCard, emptyIndex);
  //     return;
  //   }

  //   if (targetId === "board-area") {
  //     const emptyIndex = boardCards.findIndex((c) => !c);
  //     if (emptyIndex !== -1) {
  //       handleCardDropToBoard(draggedCard, emptyIndex);
  //     }
  //     return;
  //   }
  // };

  const handleDropToBoard = (card, slotIndex) => {
    clearEquities();
    setBoardCards(prev => {
      const newBoard = [...prev];
      newBoard[slotIndex] = { card, hidden: false };
      return newBoard;
    });
  };
  
  // Collect all used cards from all player hands and the board
  const usedCards = [
    ...Object.values(players)
      .flatMap(player => player.hand.filter(Boolean).map(c => c.card)), 
    ...boardCards.filter(Boolean).map(c => c.card)
  ];

  return (
    <DndContext
      sensors={sensors}                 // <-- add sensors here
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      // onDragStart={event => {
      //   console.log("Drag started:", event);
      // }}
      // onDragEnd={event => {
      //   console.log("Drag ended:", event);
      //   handleDragEnd(event)
      // }}
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
