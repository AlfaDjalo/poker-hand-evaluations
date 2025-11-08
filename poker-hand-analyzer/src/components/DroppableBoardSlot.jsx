import React from "react";
import { useDroppable } from "@dnd-kit/core";
import Card from "./Card";

const DroppableBoardSlot = ({
  index,
  cardObj,
  scale,
  showSlots,
  activeSlot,
  onCardClick,
  onSlotClick
}) => {
  const { setNodeRef, isOver } = useDroppable({
    id: `board-slot-${index}`,
    data: { index },
  });

  // React.useEffect(() => {
  //   console.log(`DroppableBoardSlot mounted: board-slot-${index}`);
  // }, [index]);

  return (
    <div
      ref={setNodeRef}
      className={`board-slot ${activeSlot === index ? 'active-slot' : ''} ${isOver ? 'hovered' : ''}`}
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

export default DroppableBoardSlot;
