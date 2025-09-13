import React from "react";
import Card from "./Card";
import "../css/BoardArea.css";

const BoardArea = ({ boardCards = [], onClick }) => {
  return (
    <div className="board-area" onClick={onClick}>
      {Array.from({ length: 50 }).map((_, index) => {
        const { card, hidden } = boardCards[index] || {};
        return (
          <div key={index} className="board-cell">
            {card && 
              <Card
                card={card}
                hidden={hidden}
                scale={1.0}
                isClickable={true}
                onClick={onClick}
              />}
          </div>
        );
      })}
    </div>
  );
};

export default BoardArea;
