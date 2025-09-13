import React from "react";
import "../css/PlayerSeat.css";
import Hand from "./Hand";

const PlayerSeat = ({ seatNumber, player, isDealer = false, isActive = false, onClick }) => {
  return (
    <div className={`player-seat ${isActive ? "active" : ""}`} onClick={onClick}>
      {player ? (
        <>
          <div className="player-name">
            {player.name} {isDealer && <span className="dealer-button">D</span>}
          </div>
          <div className="player-chips">Chips: {player.chips}</div>

          <div className="player-hand-placeholder">
            <Hand 
              cards={player.hand || []}   // ✅ use actual player.hand with { card, hidden }
              scale={0.7}
              overlap={true}
              overlapOffset={20}
            />
          </div>
        </>
      ) : (
        <div className="empty-seat">Seat {seatNumber}</div>
      )}
    </div>
  );
};

export default PlayerSeat;
