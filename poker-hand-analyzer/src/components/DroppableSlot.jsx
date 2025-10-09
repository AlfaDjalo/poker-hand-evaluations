import React from "react";
import { useDroppable } from "@dnd-kit/core";
import Card from "./Card";

const DroppableSlot = ({
  index,
  cardObj,
  variant,
  scale,
  showSlots,
  activeSlot,
  onCardClick,
  onSlotClick
}) => {
  const { setNodeRef, isOver } = useDroppable({
    id: `${variant}-slot-${index}`,
    data: { index },
  });

  React.useEffect(() => {
    console.log(`DroppableSlot mounted: ${variant}-slot-${index}`);
  }, [index]);

  return (
    <div
      ref={setNodeRef}
      className={`${variant}-slot ${activeSlot === index ? 'active-slot' : ''} ${isOver ? 'hovered' : ''}`}
    >
      <Card
        card={cardObj ? cardObj.card : null}
        scale={scale}
        isHidden={cardObj?.hidden}
        isSelected={cardObj?.selected}
        isActiveTarget={activeSlot === index}
        showOutline={showSlots}
        isClickable={true}
        // onClick={() => (cardObj ? onCardClick(index) : onSlotClick(index))}
        onClick={(e) => {
          e.stopPropagation(); // prevent BoardArea click
          if (cardObj) onCardClick(index);
          else onSlotClick(index);
        }}
      />
    </div>
  );
};

export default DroppableSlot;
