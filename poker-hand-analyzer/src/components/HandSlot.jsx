// HandSlot.jsx
import React from "react";
import { useDroppable } from "@dnd-kit/core";
import Card from "./Card";

const HandSlot = ({ card, seat, slotIndex, onCardClick, onSlotClick }) => {
  const { setNodeRef, isOver } = useDroppable({
    id: `player-${seat}-slot-${slotIndex}`,
    data: { seat, slotIndex },
  });

  return (
    <div
      ref={setNodeRef}
      className={`hand-slot ${isOver ? "hovered" : ""}`}
      onClick={() => onSlotClick(seat, slotIndex)}
      style={{
        width: 60,
        height: 90,
        border: "1px dashed gray",
        margin: 4,
        position: "relative",
      }}
    >
      {card && (
        <Card
          card={card}
          scale={1}
          isClickable={true}
          onClick={() => onCardClick(seat, slotIndex)}
        />
      )}
    </div>
  );
};

export default HandSlot;
