import React, { useState } from "react";
import PlayerSeat from "./PlayerSeat";
import BoardArea from "./BoardArea";
import "../css/PokerTable.css";

const PokerTable = ({ players, boardCards, dealerSeat, activeTarget, onSeatClick, onBoardClick }) => {

  return (
    <div className="poker-table-container">
      <div className="poker-table">
        {/* Outer ellipse - green felt */}
        <div className="table-outer">
          {/* Inner ellipse - betting line */}
          <div
            className={`table-inner ${activeTarget?.type === "board" ? "active" : ""}`}
            onClick={onBoardClick}
          >
            {/* Board area with 5x10 grid */}
            <div className="board-container">
              <BoardArea boardCards={boardCards} onClick={onBoardClick}/>
            </div>
          </div>
        </div>

        {/* Seats positioned around the ellipse */}
        {Array.from({ length: 6 }, (_, i) => {
          const seatNum = i + 1;
          return (
            <div key={seatNum} className={`seat seat-${seatNum}`}>
              <PlayerSeat
                seat={seatNum}
                player={players[seatNum]}
                isDealer={dealerSeat === seatNum}
                isActive={activeTarget?.type === "player" && activeTarget.seat === seatNum}
                onClick={() => onSeatClick?.(seatNum)}
              />
            </div>
          );
        })}
      </div>
    </div>
  );

  //   <div className="poker-table">
  //     {/* Outer ellipse */}
  //     <div className="table-outer">
  //       {/* Inner ellipse (betting line) */}
  //       <div
  //         className={`table-inner ${activeTarget?.type === "board" ? "active" : ""}`}
  //         onClick={onBoardClick}
  //       >
  //         <BoardArea boardCards={boardCards} />
  //       </div>

  //       {/* Seats */}
  //       {Array.from({ length: 6 }, (_, i) => {
  //         const seatNum = i + 1;
  //         return (
  //           <div key={seatNum} className={`seat seat-${seatNum}`}>
  //             <PlayerSeat
  //               seat={seatNum}
  //               player={players[seatNum]}
  //               isDealer={dealerSeat === seatNum}
  //               isActive={activeTarget?.type === "player" && activeTarget.seat === seatNum}
  //               onClick={() => onSeatClick(seatNum)}
  //             />
  //           </div>
  //         );
  //       })}
  //     </div>
  //   </div>
  // );
};

export default PokerTable;
