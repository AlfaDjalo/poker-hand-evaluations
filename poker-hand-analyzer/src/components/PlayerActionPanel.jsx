import React, { useState } from 'react';
import "../css/PlayerActionPanel.css";

const PlayerActionPanel = ({ player, availableActions, onAction }) => {
    const [betAmount, setBetAmount] = useState(0);

    const isAvailable = (action) => availableActions.includes(action);

    console.log("Available actions for", player?.name, availableActions);
    console.log("isAvailable", isAvailable);

    return (
        <div className="player-action-panel">
        <button
            className={`fold ${!isAvailable("fold") ? "disabled" : ""}`}
            onClick={() => isAvailable("fold") && onAction("fold")}
        >
            Fold
        </button>

        <button
            className={`call ${!isAvailable("call") || isAvailable("check") ? "disabled" : ""}`}
            onClick={() => {
                if (isAvailable("call")) onAction("call");
                else if (isAvailable("check")) onAction("check");
            }}
        >
            { isAvailable("call") ? "Call" : isAvailable("check") ? "Check" : "-" } 
        </button>

        <div className="bet-section">
            <input
            type="number"
            value={betAmount}
            onChange={(e) => setBetAmount(e.target.value)}
            placeholder="Bet size"
            />
            <button
            className={`bet ${!isAvailable("bet") ? "disabled" : ""}`}
            onClick={() =>
                isAvailable("bet") && onAction("bet", parseFloat(betAmount))
            }
            >
            {player?.contributionCurrentStreet > 0 ? "Raise" : "Bet"}
            </button>
        </div>
        </div>
    );
};

export default PlayerActionPanel;
