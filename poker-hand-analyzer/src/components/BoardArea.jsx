import React from "react";
import DroppableSlot from "./DroppableSlot";
// import DroppableBoardSlot from "./DroppableBoardSlot";
import { useDroppable } from "@dnd-kit/core";
import "../css/BoardArea.css";

const BoardArea = ({ 
  cards,
  scale,
  showSlots,
  activeSlot, 
  onCardClick,
  onSlotClick, 
  onAreaClick,
  maxCards=5
 }) => {
  // Create array with empty slots
  const slots = Array.from({ length: maxCards }, (_, index) => 
    cards[index] || null
  );

  const { setNodeRef: setBoardAreaRef, isOver: isOverBoardArea } = useDroppable({
    id: "board-area",
    data: "board-area",
    // data: { type: "board-area" },
  });

  // const { setNodeRef, isOver } = useDroppable({
  //   id: "board-area",
  //   data: { type: "board-area" },
  //   disabled: cards.some(Boolean), // disable if there are slots/cards
  // });

  React.useEffect(() => {
    console.log("BoardArea droppable mounted: board-area");
  }, []);

  return (
    // <div className="board-area" onClick={onAreaClick}>

    <div
      ref={setBoardAreaRef}
      className={`board-area ${isOverBoardArea ? "hovered" : ""}`}
      onClick={(e) => {
        // Detect clicks *not* on any child slot
        if (e.target === e.currentTarget) {
          onAreaClick();
        }
      }}
    >
      {slots.map((cardObj, index) => (
        <DroppableSlot
        // <DroppableBoardSlot
          key={index}
          index={index}
          variant={"board"}
          cardObj={cardObj}
          scale={scale}
          showSlots={showSlots}
          activeSlot={activeSlot}
          onCardClick={onCardClick}
          onSlotClick={onSlotClick}
        />
      ))}
    </div>
  );
};

export default BoardArea;
