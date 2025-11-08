// GameConfig.jsx
import React, { useState } from 'react';

const GameConfig = ({ onSave }) => {
  const [smallBlind, setSmallBlind] = useState(1);
  const [bigBlind, setBigBlind] = useState(2);
  const [startingStack, setStartingStack] = useState(2000);

  return (
    <div>
      <h3>Game Configuration</h3>
      <label>Small Blind</label>
      <input value={smallBlind} onChange={e => setSmallBlind(Number(e.target.value))} />
      <label>Big Blind</label>
      <input value={bigBlind} onChange={e => setBigBlind(Number(e.target.value))} />
      <label>Starting Stack</label>
      <input value={startingStack} onChange={e => setStartingStack(Number(e.target.value))} />
      <button onClick={() => onSave({ blinds: [smallBlind, bigBlind], startingStack })}>
        Save
      </button>
    </div>
  );
};
