import React from "react";
// import { useDroppable } from "@dnd-kit/core";
// import Card from "./Card";
import DroppableBoardSlot from "./DroppableBoardSlot";
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
  onDropToBoard,
  maxCards=5
 }) => {
  // Create array with empty slots
  const slots = Array.from({ length: maxCards }, (_, index) => 
    cards[index] || null
  );

  // handle click on the container
  // const handleBoardClick = (e) => {
  //   onAreaClick()
    // if (e.target === e.currentTarget) {
      // click on the empty space, not a card
      // onSelectBoard();
    // }
  // };

  const { setNodeRef, isOver } = useDroppable({
    id: "board-area",
    data: { type: "board-area" },
  });
  
  React.useEffect(() => {
    console.log("BoardArea droppable mounted: board-area");
  }, []);

  return (
    // <div className="board-area" onClick={onAreaClick}>

    <div
      ref={setNodeRef}
      className={`board-area ${isOver ? "hovered" : ""}`}
      onClick={onAreaClick}
    >

      {slots.map((cardObj, index) => (
        <DroppableBoardSlot
          key={index}
          index={index}
          cardObj={cardObj}
          scale={scale}
          showSlots={showSlots}
          activeSlot={activeSlot}
          onCardClick={onCardClick}
          onSlotClick={onSlotClick}
        />
        // const { setNodeRef, isOver } = useDroppable({
        //   id: `board-slot-${index}`,
        //   data: { index },
        // });

        // return (
        //   <div
        //     ref={setNodeRef}
        //     key={index}
        //     className={`board-slot ${activeSlot === index ? "active-slot" : ""} ${
        //       isOver ? "hovered" : ""
        //     }`}
        //   >          

        //     <Card
        //       card={cardObj ? cardObj.card : null}
        //       scale={scale}
        //       isHidden={cardObj?.hidden}
        //       isSelected={cardObj?.selected}
        //       isActiveTarget={activeSlot === index}
        //       showOutline={showSlots}
        //       isClickable={true}
        //       onClick={() => (cardObj ? onCardClick(index) : onSlotClick(index))}
        //     />
        //   </div>
        // );
      ))}
    </div>
  );
};

export default BoardArea;
