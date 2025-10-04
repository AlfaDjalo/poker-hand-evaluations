import React, { useState } from "react";
import PlayerSeat from "./PlayerSeat";
import BoardArea from "./BoardArea";
import "../css/PokerTable.css";

const PokerTable = ({ 
  players,
  boardCards,
  dealerSeat,
  activeTarget,
  onSeatClick,
  onPlayerCardClick,
  onPlayerSlotClick,
  onBoardCardClick,
  onBoardSlotClick,
  onBoardAreaClick,
  loading
}) => {
  
  return (
    <div className="poker-table-container">
      <div className="poker-table">
        {/* Outer ellipse - green felt */}
        <div className="table-outer">
          {/* Inner ellipse - betting line */}
          <div
            className={`table-inner ${activeTarget?.type === "board" ? "active" : ""}`}
            onClick={onBoardAreaClick}
          >
            <div className="board-container">
              <BoardArea
                cards={boardCards}
                scale={1.0}
                showSlots={true}
                activeSlot={activeTarget?.type === "board" ? activeTarget.slot : null}
                onSlotClick={onBoardSlotClick}
                onCardClick={onBoardCardClick}
                onAreaClick={onBoardAreaClick}
                maxCards={5}
              />
            </div>
          </div>
        </div>

        {Array.from({ length: 6 }, (_, i) => {
          const seatNum = i + 1;
          return (
            <div key={seatNum} className={`seat seat-${seatNum}`}>
              <PlayerSeat
                // key={seatNum}
                seatNumber={seatNum}
                player={players[seatNum]}
                isActive={activeTarget?.type === "player" && activeTarget.seat === seatNum}
                activeSlot={activeTarget?.type === "player" && activeTarget.seat === seatNum ? activeTarget.slot : null}
                onCardClick={(cardIndex) => onPlayerCardClick(seatNum, cardIndex)}
                onSlotClick={(slotIndex) => onPlayerSlotClick(seatNum, slotIndex)}
                onSeatClick={() => onSeatClick(seatNum)}
                loading={loading}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default PokerTable;
