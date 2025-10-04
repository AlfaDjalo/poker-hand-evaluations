import React from "react";
import Card from "./Card";
import "../css/Hand.css";

const Hand = ({ 
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

  return (
    <div className="hand">
      {slots.map((cardObj, index) => (
        <div 
          key={index} 
          className={`hand-slot ${activeSlot === index ? 'active-slot' : ''}`}
        >
          <Card
            card={cardObj ? cardObj.card : null}
            scale={scale}
            isHidden={cardObj?.hidden}
            isSelected={cardObj?.selected}
            isActiveTarget={activeSlot === index} 
            showOutline={showSlots}
            isClickable={true}
            onClick={() => (cardObj ? onCardClick(index) : onSlotClick(index))}
          />
        </div>
      ))}
    </div>
  );
};

export default Hand;