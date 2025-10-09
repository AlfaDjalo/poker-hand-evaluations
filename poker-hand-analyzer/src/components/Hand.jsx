import React from "react";
// import Card from "./Card";
import "../css/Hand.css";
import DroppableSlot from "./DroppableSlot";
// import { useDroppable } from "@dnd-kit/core";

const Hand = ({ 
  seatNumber,
  cards, 
  scale, 
  showSlots, 
  activeSlot, 
  onCardClick, 
  onSlotClick, 
  maxCards = 4
}) => {
  // Create array with empty slots
  const slots = Array.from({ length: maxCards }, (_, index) => 
    cards[index] || null
  );

  // const { setNodeRef: setHandAreaRef, isOver: isOverHandArea } = useDroppable({
  //   id: "player-hand-area",
  //   data: "player-hand-area",
  //   // data: { type: "board-area" },
  // });

  return (
    <div className="hand">
      {slots.map((cardObj, index) => (
        <DroppableSlot
        // <DroppableBoardSlot
          key={index}
          index={index}
          variant={`player-${seatNumber}`}
          cardObj={cardObj}
          scale={scale}
          showSlots={showSlots}
          activeSlot={activeSlot}
          onCardClick={onCardClick}
          onSlotClick={onSlotClick}
        />
        // <div 
        //   key={index} 
        //   className={`hand-slot ${activeSlot === index ? 'active-slot' : ''}`}
        // >
        //   <Card
        //     card={cardObj ? cardObj.card : null}
        //     scale={scale}
        //     isHidden={cardObj?.hidden}
        //     isSelected={cardObj?.selected}
        //     isActiveTarget={activeSlot === index} 
        //     showOutline={showSlots}
        //     isClickable={true}
        //     onClick={() => (cardObj ? onCardClick(index) : onSlotClick(index))}
        //   />
        // </div>
      ))}
    </div>
  );
};

export default Hand;